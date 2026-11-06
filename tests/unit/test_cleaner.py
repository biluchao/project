"""
File Path : tests/unit/test_cleaner.py
Module Name : test_cleaner
Purpose : DataCleaner / TimeAligner / compute_quality_score 单元测试
Author : system
Created : 2026-11-06
Modified : 2026-11-06
Version : 1.0.0
Standard : QCS v1.0 / FPS v2.0
Dependencies : pytest, src.data.loader, src.data.cleaner
Capabilities : 覆盖OHLC/重复/对齐/跳空/零量/时间缺口/质量分
HMAC Required : no
Immutable : no
Max Lines : 0/800
Static Check : py_compile, isort, black, flake8, mypy, radon
Change Log : 2026-11-06 初始生成
"""
from __future__ import annotations

import pytest

from src.data.cleaner import (
    CleanerConfig,
    DataCleaner,
    TimeAligner,
    compute_quality_score,
)
from src.data.loader import (
    CleanBar,
    DataLoadError,
    RawBar,
    FLAG_GAP_ANOMALY,
    FLAG_TIME_GAP,
    FLAG_ZERO_VOLUME,
)

INTERVAL = 180_000


def make_bar(
    ts: int,
    o: float = 100.0,
    h: float = 105.0,
    l: float = 95.0,
    c: float = 100.0,
    v: float = 1000.0,
) -> RawBar:
    """测试用 RawBar 工厂"""
    return RawBar(
        timestamp=ts, open=o, high=h, low=l, close=c, volume=v,
    )


class TestDataCleaner:
    def test_valid_bar_passes(self) -> None:
        """合法K线通过，flags=0"""
        cleaner = DataCleaner(CleanerConfig(interval_ms=INTERVAL))
        raw = [make_bar(INTERVAL)]
        cleaned, issues = cleaner.clean(raw)
        assert len(cleaned) == 1
        assert len(issues) == 0
        assert cleaned[0].flags == 0

    def test_ohlc_invalid_discarded(self) -> None:
        """H < L 被丢弃"""
        cleaner = DataCleaner(CleanerConfig(interval_ms=INTERVAL))
        raw = [make_bar(INTERVAL, o=100, h=90, l=95, c=100)]
        cleaned, issues = cleaner.clean(raw)
        assert len(cleaned) == 0
        assert len(issues) == 1
        assert issues[0].issue_type == "OHLC_INVALID"

    def test_zero_price_discarded(self) -> None:
        """close <= min_price 被丢弃"""
        cleaner = DataCleaner(CleanerConfig(interval_ms=INTERVAL))
        raw = [make_bar(INTERVAL, o=0.0, h=105.0, l=0.0, c=0.0)]
        cleaned, issues = cleaner.clean(raw)
        assert len(cleaned) == 0
        assert issues[0].issue_type == "ZERO_PRICE"

    def test_duplicate_timestamp_keeps_first(self) -> None:
        """时间戳重复只保留第一条"""
        cleaner = DataCleaner(CleanerConfig(interval_ms=INTERVAL))
        raw = [
            make_bar(INTERVAL, c=100),
            make_bar(INTERVAL, o=100, h=205, l=95, c=200),
        ]
        cleaned, issues = cleaner.clean(raw)
        assert len(cleaned) == 1
        assert cleaned[0].close == 100
        assert issues[0].issue_type == "DUPLICATE"

    def test_misaligned_timestamp_discarded(self) -> None:
        """时间戳不对齐被丢弃"""
        cleaner = DataCleaner(CleanerConfig(interval_ms=INTERVAL))
        raw = [make_bar(1000)]
        cleaned, issues = cleaner.clean(raw)
        assert len(cleaned) == 0
        assert issues[0].issue_type == "MISALIGNED"

    def test_gap_anomaly_flagged(self) -> None:
        """跳空 > max_price_gap 标记 FLAG_GAP_ANOMALY"""
        cleaner = DataCleaner(
            CleanerConfig(interval_ms=INTERVAL, max_price_gap=0.10)
        )
        raw = [
            make_bar(INTERVAL, o=100, h=105, l=95, c=100),
            make_bar(2 * INTERVAL, o=120, h=125, l=118, c=120),
        ]
        cleaned, _ = cleaner.clean(raw)
        assert len(cleaned) == 2
        assert cleaned[1].flags & FLAG_GAP_ANOMALY

    def test_zero_volume_flagged(self) -> None:
        """零成交量标记 FLAG_ZERO_VOLUME"""
        cleaner = DataCleaner(
            CleanerConfig(interval_ms=INTERVAL, require_volume=True)
        )
        raw = [make_bar(INTERVAL, v=0.0)]
        cleaned, _ = cleaner.clean(raw)
        assert cleaned[0].flags & FLAG_ZERO_VOLUME

    def test_time_gap_flagged(self) -> None:
        """时间不连续标记 FLAG_TIME_GAP"""
        cleaner = DataCleaner(CleanerConfig(interval_ms=INTERVAL))
        raw = [
            make_bar(INTERVAL),
            make_bar(4 * INTERVAL),
        ]
        cleaned, _ = cleaner.clean(raw)
        assert cleaned[1].flags & FLAG_TIME_GAP


class TestTimeAligner:
    def test_align_no_gap(self) -> None:
        """连续3根，无缺失"""
        cleaner = DataCleaner(CleanerConfig(interval_ms=INTERVAL))
        raw = [make_bar(i * INTERVAL) for i in range(1, 4)]
        cleaned, _ = cleaner.clean(raw)
        aligner = TimeAligner()
        aligned = aligner.align(cleaned, INTERVAL, "skip")
        assert len(aligned) == 3
        assert all(b is not None for b in aligned)

    def test_align_with_gap_skip(self) -> None:
        """中间缺失，skip 模式留 None"""
        cleaner = DataCleaner(CleanerConfig(interval_ms=INTERVAL))
        raw = [make_bar(INTERVAL), make_bar(3 * INTERVAL)]
        cleaned, _ = cleaner.clean(raw)
        aligner = TimeAligner()
        aligned = aligner.align(cleaned, INTERVAL, "skip")
        assert len(aligned) == 3
        assert aligned[0] is not None
        assert aligned[1] is None
        assert aligned[2] is not None

    def test_align_with_gap_fill_prev_close(self) -> None:
        """中间缺失，fill_prev_close 模式填合成K线"""
        cleaner = DataCleaner(CleanerConfig(interval_ms=INTERVAL))
        raw = [
            make_bar(INTERVAL, c=100),
            make_bar(3 * INTERVAL, o=110, h=115, l=105, c=110),
        ]
        cleaned, _ = cleaner.clean(raw)
        aligner = TimeAligner()
        aligned = aligner.align(cleaned, INTERVAL, "fill_prev_close")
        assert len(aligned) == 3
        synthetic_bar = aligned[1]
        assert synthetic_bar is not None
        assert synthetic_bar.synthetic is True
        assert synthetic_bar.close == 100

    def test_align_with_gap_abort(self) -> None:
        """中间缺失，abort 模式抛异常"""
        cleaner = DataCleaner(CleanerConfig(interval_ms=INTERVAL))
        raw = [make_bar(INTERVAL), make_bar(3 * INTERVAL)]
        cleaned, _ = cleaner.clean(raw)
        aligner = TimeAligner()
        with pytest.raises(DataLoadError) as exc_info:
            aligner.align(cleaned, INTERVAL, "abort")
        assert exc_info.value.code == "GAP_ABORT"


class TestQualityScore:
    def test_all_valid(self) -> None:
        """全部合法，score = 1.0"""
        aligned: list[CleanBar | None] = [
            CleanBar(
                timestamp=i * INTERVAL,
                open=100.0, high=105.0, low=95.0, close=100.0,
                volume=1000.0, raw_index=i, flags=0,
            )
            for i in range(10)
        ]
        assert compute_quality_score(aligned) == 1.0

    def test_half_missing(self) -> None:
        """一半缺失，score = 0.5"""
        aligned: list[CleanBar | None] = [None] * 10
        for i in range(5):
            aligned[i] = CleanBar(
                timestamp=i * INTERVAL,
                open=100.0, high=105.0, low=95.0, close=100.0,
                volume=1000.0, raw_index=i, flags=0,
            )
        assert compute_quality_score(aligned) == 0.5

    def test_empty(self) -> None:
        """空列表返回 0"""
        assert compute_quality_score([]) == 0.0
