"""
File Path : src/data/loader.py
Module Name : loader
Purpose : 从币安CSV加载K线数据，转为RawBar
Author : system
Created : 2026-11-06
Modified : 2026-11-06
Version : 1.0.0
Standard : QCS v1.0 / FPS v2.0
Dependencies : csv, dataclasses, pathlib, typing
Capabilities : BinanceCSVSource.load, RawBar, CleanBar, DataIssue
HMAC Required : no
Immutable : partial (RawBar/DataIssue frozen)
Max Lines : 0/800
Static Check : py_compile, isort, black, flake8, mypy, radon
Change Log : 2026-11-06 初始生成
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ── 标志位（v8.0.0 第2部分）──
FLAG_OHLC_INVALID = 1 << 0
FLAG_ZERO_VOLUME = 1 << 1
FLAG_TIME_GAP = 1 << 2
FLAG_GAP_ANOMALY = 1 << 3
FLAG_DUPLICATE = 1 << 4


@dataclass(frozen=True)
class RawBar:
    """CSV读取的原始K线，未做任何清洗"""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int = 0


@dataclass
class CleanBar:
    """清洗后的K线。flags 位标记异常"""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    raw_index: int
    flags: int = 0
    synthetic: bool = False


@dataclass(frozen=True)
class DataIssue:
    """单行数据的异常记录"""

    raw_index: int
    issue_type: str
    detail: str
    raw_bar: Optional[RawBar] = None


class DataLoadError(Exception):
    """数据加载失败"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class BinanceCSVSource:
    """币安K线CSV数据源

    格式：open_time,open,high,low,close,volume,close_time,...
    第一行为表头时自动跳过
    """

    HEADER_FIRST_COL = "open_time"
    MIN_COLUMNS = 6

    def load(self, path: str | Path) -> list[RawBar]:
        """加载CSV，返回RawBar列表。失败时抛 DataLoadError"""
        p = Path(path)
        if not p.exists():
            raise DataLoadError("FILE_NOT_FOUND", f"file not found: {p}")

        bars: list[RawBar] = []
        try:
            with p.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                for idx, row in enumerate(reader):
                    if idx == 0 and row and row[0] == self.HEADER_FIRST_COL:
                        continue
                    bar = self._parse_row(row)
                    if bar is not None:
                        bars.append(bar)
        except OSError as e:
            raise DataLoadError("READ_ERROR", f"read failed: {e}") from e

        if not bars:
            raise DataLoadError("EMPTY_DATA", f"no valid bars in {p}")
        return bars

    def _parse_row(self, row: list[str]) -> Optional[RawBar]:
        """解析单行。字段不足或数值非法时返回 None"""
        if len(row) < self.MIN_COLUMNS:
            return None
        try:
            return RawBar(
                timestamp=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                close_time=int(row[6]) if len(row) > 6 else 0,
            )
        except (ValueError, IndexError):
            return None
