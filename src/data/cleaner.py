"""
File Path : src/data/cleaner.py
Module Name : cleaner
Purpose : K线清洗、时间对齐、质量评分
Author : system
Created : 2026-11-06
Modified : 2026-11-06
Version : 1.0.0
Standard : QCS v1.0 / FPS v2.0
Dependencies : .loader (RawBar/CleanBar/DataIssue/flags), time
Capabilities : DataCleaner.clean, TimeAligner.align, compute_quality_score, build_manifest
HMAC Required : no
Immutable : partial (CleanerConfig/DataManifest frozen)
Max Lines : 0/800
Static Check : py_compile, isort, black, flake8, mypy, radon
Change Log : 2026-11-06 初始生成
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from .loader import (
    CleanBar,
    DataIssue,
    DataLoadError,
    RawBar,
    FLAG_DUPLICATE,
    FLAG_GAP_ANOMALY,
    FLAG_OHLC_INVALID,
    FLAG_TIME_GAP,
    FLAG_ZERO_VOLUME,
)


@dataclass(frozen=True)
class CleanerConfig:
    """清洗配置（对应 config.yaml data 模块）"""

    interval_ms: int = 180_000
    min_price: float = 1e-8
    max_price_gap: float = 0.10
    require_volume: bool = True
    gap_policy: str = "skip"  # skip / fill_prev_close / abort
    min_quality_score: float = 0.90


@dataclass(frozen=True)
class DataManifest:
    """数据质量清单"""

    symbol: str
    interval_sec: int
    start_ts: int
    end_ts: int
    bar_count_total: int
    bar_count_valid: int
    bar_count_flagged: int
    bar_count_synthetic: int
    bar_count_missing: int
    gap_count: int
    gap_total_bars: int
    quality_score: float
    gap_policy: str
    loaded_at: int


class DataCleaner:
    """K线清洗管道

    规则（v8.0.0 第2部分）：
      1. OHLC 非法 → 丢弃（不修复）
      2. 零/负价格 → 丢弃
      3. 时间戳重复 → 保留第一条
      4. 时间戳不对齐 → 丢弃
      5. 跳空 > max_price_gap → 标记（open vs prev_close）
      6. 零成交量 → 标记
      7. 时间不连续 → 标记
    """

    def __init__(self, config: CleanerConfig) -> None:
        self._config = config

    def clean(
        self, raw: list[RawBar]
    ) -> tuple[list[CleanBar], list[DataIssue]]:
        """清洗原始K线，返回 (清洗后列表, 问题列表)"""
        cleaned: list[CleanBar] = []
        issues: list[DataIssue] = []
        seen_ts: set[int] = set()

        for idx, bar in enumerate(raw):
            # Step 1 时间戳重复
            if bar.timestamp in seen_ts:
                issues.append(DataIssue(
                    raw_index=idx,
                    issue_type="DUPLICATE",
                    detail=f"ts={bar.timestamp}",
                    raw_bar=bar,
                ))
                continue
            seen_ts.add(bar.timestamp)

            # Step 2 时间戳对齐
            if bar.timestamp % self._config.interval_ms != 0:
                issues.append(DataIssue(
                    raw_index=idx,
                    issue_type="MISALIGNED",
                    detail=f"ts={bar.timestamp} % {self._config.interval_ms}",
                    raw_bar=bar,
                ))
                continue

            # Step 3 OHLC 合法性
            if not (
                bar.low <= bar.open <= bar.high
                and bar.low <= bar.close <= bar.high
            ):
                issues.append(DataIssue(
                    raw_index=idx,
                    issue_type="OHLC_INVALID",
                    detail=(
                        f"O={bar.open} H={bar.high} "
                        f"L={bar.low} C={bar.close}"
                    ),
                    raw_bar=bar,
                ))
                continue

            # Step 4 价格必须 > min_price
            if bar.close <= self._config.min_price:
                issues.append(DataIssue(
                    raw_index=idx,
                    issue_type="ZERO_PRICE",
                    detail=f"C={bar.close}",
                    raw_bar=bar,
                ))
                continue

            # Step 5 标记异常（不丢弃）
            flags = 0
            if self._config.require_volume and bar.volume <= 0:
                flags |= FLAG_ZERO_VOLUME

            if cleaned:
                prev = cleaned[-1]
                # 跳空：open vs prev_close（v8.0.0 规定）
                gap = abs(bar.open - prev.close) / max(
                    prev.close, self._config.min_price
                )
                if gap > self._config.max_price_gap:
                    flags |= FLAG_GAP_ANOMALY

                # 时间不连续
                expected = prev.timestamp + self._config.interval_ms
                if bar.timestamp != expected:
                    flags |= FLAG_TIME_GAP

            cleaned.append(CleanBar(
                timestamp=bar.timestamp,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                raw_index=idx,
                flags=flags,
            ))

        return cleaned, issues


class TimeAligner:
    """时间对齐。缺失位置按 gap_policy 处理"""

    def align(
        self,
        bars: list[CleanBar],
        interval_ms: int,
        gap_policy: str,
    ) -> list[Optional[CleanBar]]:
        """返回对齐后的序列。None 表示缺失（skip 模式）"""
        if not bars:
            return []

        start = bars[0].timestamp
        end = bars[-1].timestamp
        count = (end - start) // interval_ms + 1

        aligned: list[Optional[CleanBar]] = [None] * count
        idx_map = {start + i * interval_ms: i for i in range(count)}

        for bar in bars:
            idx = idx_map.get(bar.timestamp)
            if idx is not None:
                aligned[idx] = bar

        missing_count = sum(1 for b in aligned if b is None)
        if gap_policy == "abort" and missing_count > 0:
            raise DataLoadError(
                "GAP_ABORT",
                f"found {missing_count} missing bars, abort policy",
            )

        if gap_policy == "fill_prev_close":
            prev_close = bars[0].close
            for i in range(count):
                bar_at_i = aligned[i]
                if bar_at_i is None:
                    aligned[i] = CleanBar(
                        timestamp=start + i * interval_ms,
                        open=prev_close,
                        high=prev_close,
                        low=prev_close,
                        close=prev_close,
                        volume=0.0,
                        raw_index=-1,
                        flags=FLAG_TIME_GAP,
                        synthetic=True,
                    )
                else:
                    prev_close = bar_at_i.close

        return aligned


def compute_quality_score(
    aligned: list[Optional[CleanBar]],
) -> float:
    """计算数据质量分（v6.9.0 公式）

    valid=1.0, flagged=0.5, synthetic=0.3, missing=0.0
    """
    total = len(aligned)
    if total == 0:
        return 0.0

    valid = 0
    flagged = 0
    synthetic = 0
    for bar in aligned:
        if bar is None:
            continue
        if bar.synthetic:
            synthetic += 1
        elif bar.flags == 0:
            valid += 1
        else:
            flagged += 1

    missing = total - valid - flagged - synthetic
    score = (
        valid * 1.0
        + flagged * 0.5
        + synthetic * 0.3
        + missing * 0.0
    ) / total
    return round(score, 4)


def build_manifest(
    symbol: str,
    interval_ms: int,
    raw: list[RawBar],
    aligned: list[Optional[CleanBar]],
    quality_score: float,
    gap_policy: str,
) -> DataManifest:
    """构建数据质量清单"""
    valid = 0
    flagged = 0
    synthetic = 0
    missing = 0
    for bar in aligned:
        if bar is None:
            missing += 1
        elif bar.synthetic:
            synthetic += 1
        elif bar.flags == 0:
            valid += 1
        else:
            flagged += 1

    gap_count = 0
    gap_total_bars = 0
    prev_ts = 0
    for bar in aligned:
        if bar is None:
            continue
        if prev_ts > 0:
            delta = (bar.timestamp - prev_ts) // interval_ms
            if delta > 1:
                gap_count += 1
                gap_total_bars += delta - 1
        prev_ts = bar.timestamp

    return DataManifest(
        symbol=symbol,
        interval_sec=interval_ms // 1000,
        start_ts=raw[0].timestamp if raw else 0,
        end_ts=raw[-1].timestamp if raw else 0,
        bar_count_total=len(raw),
        bar_count_valid=valid,
        bar_count_flagged=flagged,
        bar_count_synthetic=synthetic,
        bar_count_missing=missing,
        gap_count=gap_count,
        gap_total_bars=gap_total_bars,
        quality_score=quality_score,
        gap_policy=gap_policy,
        loaded_at=int(time.time() * 1000),
    )
