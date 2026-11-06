#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从币安官方数据仓库下载 2024 年 BTCUSDT 3 分钟 K 线，
合并、清洗并保存为 data/BTCUSDT_3m_2024.csv。

数据源: https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/3m/

运行:
    pip install requests pandas
    python fetch_btc_3m_2024.py

输出:
    data/BTCUSDT_3m_2024.csv

可选:
    - 原始 ZIP 会缓存到 ./cache/，重复运行不会重复下载
    - 删除 cache/ 目录可强制重新下载
"""

import io
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ================= 配置 =================
SYMBOL = "BTCUSDT"
INTERVAL = "3m"                 # 3 分钟 K 线
YEAR = 2024
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"

# 输出目录与文件（相对项目根目录）
OUTPUT_DIR = Path("data")
OUTPUT_CSV = OUTPUT_DIR / f"{SYMBOL}_{INTERVAL}_{YEAR}.csv"

CACHE_DIR = Path("cache")
USE_CACHE = True                # 是否缓存原始 ZIP

# 币安公开数据 CSV 的列（无表头）
COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
]

# 365 天 × 24 小时 × 20 根/小时
EXPECTED_ROWS = 365 * 24 * 20


def build_session() -> requests.Session:
    """带重试的 requests session"""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "btc-3m-fetcher/1.0"})
    return session


def download_month(session: requests.Session, month: int):
    """下载某个月的 ZIP，返回 (bytes, zip_name)；不存在返回 (None, zip_name)"""
    month_str = f"{month:02d}"
    zip_name = f"{SYMBOL}-{INTERVAL}-{YEAR}-{month_str}.zip"
    url = f"{BASE_URL}/{SYMBOL}/{INTERVAL}/{zip_name}"

    cache_path = CACHE_DIR / zip_name
    if USE_CACHE and cache_path.exists() and cache_path.stat().st_size > 0:
        print(f"  [cache] {month_str} 月命中缓存")
        return cache_path.read_bytes(), zip_name

    print(f"  下载 {month_str} 月: {url}")
    resp = session.get(url, timeout=60)
    if resp.status_code == 404:
        print(f"  [warn] {month_str} 月文件不存在（可能无数据），跳过")
        return None, zip_name
    resp.raise_for_status()

    if USE_CACHE:
        CACHE_DIR.mkdir(exist_ok=True)
        cache_path.write_bytes(resp.content)
    return resp.content, zip_name


def parse_zip(content: bytes, zip_name: str) -> pd.DataFrame:
    """从 ZIP 内容中读取 CSV"""
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"{zip_name} 内未找到 CSV 文件")
        with z.open(csv_names[0]) as f:
            df = pd.read_csv(f, header=None, names=COLUMNS)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """清洗、类型转换、排序"""
    df = df.drop(columns=["ignore"], errors="ignore")

    # 去重
    before = len(df)
    df = df.drop_duplicates(subset=["open_time"]).reset_index(drop=True)
    if len(df) != before:
        print(f"  去重: {before:,} -> {len(df):,}")

    # 排序
    df = df.sort_values("open_time").reset_index(drop=True)

    # 时间转换（毫秒 -> UTC datetime）
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # 数值类型
    float_cols = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
    ]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["number_of_trades"] = (
        pd.to_numeric(df["number_of_trades"], errors="coerce").astype("Int64")
    )
    return df


def validate(df: pd.DataFrame) -> None:
    """基本校验，打印摘要"""
    print("\n==> 校验")
    print(f"  行数: {len(df):,}  (预期约 {EXPECTED_ROWS:,})")
    print(f"  时间范围: {df['open_time'].min()} ~ {df['open_time'].max()}")
    print(f"  字段: {list(df.columns)}")
    print("\n  前 3 行:")
    print(df[["open_time", "open", "high", "low", "close", "volume"]].head(3).to_string(index=False))
    print("\n  后 3 行:")
    print(df[["open_time", "open", "high", "low", "close", "volume"]].tail(3).to_string(index=False))

    if len(df) < EXPECTED_ROWS * 0.99:
        print(
            f"\n  [warn] 行数({len(df):,})明显少于预期({EXPECTED_ROWS:,})，"
            f"可能有月份缺失或数据源延迟"
        )


def main() -> None:
    print(f"==> 拉取 {SYMBOL} {INTERVAL} {YEAR} 全年数据")
    session = build_session()

    all_dfs = []
    for month in range(1, 13):
        content, zip_name = download_month(session, month)
        if content is None:
            continue
        df = parse_zip(content, zip_name)
        all_dfs.append(df)
        total = sum(len(d) for d in all_dfs)
        print(f"  {month:02d} 月 {len(df):,} 行，累计 {total:,} 行")
        time.sleep(0.3)  # 礼貌节流

    if not all_dfs:
        print("ERROR: 未获取到任何数据，请检查网络或参数。")
        sys.exit(1)

    print("\n==> 合并清洗")
    df = pd.concat(all_dfs, ignore_index=True)
    df = clean(df)

    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"==> 保存到 {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"==> 完成: {OUTPUT_CSV.resolve()}")

    validate(df)


if __name__ == "__main__":
    main()
