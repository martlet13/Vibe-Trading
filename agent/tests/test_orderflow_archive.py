"""Bybit trade-archive order flow attached to backtest bars.

Archive files are built in memory in both published formats; no test opens a
socket. The runtime root points at ``tmp_path`` so the cache is isolated.
"""

from __future__ import annotations

import gzip
from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtest import orderflow_archive as oa
from backtest.engines.base import _maybe_enrich_orderflow
from backtest.runner import _unrecognized_config_keys

DAY = date(2026, 11, 16)
EPOCH = pd.Timestamp("2026-11-16").timestamp()


def _perp_file(rows):
    lines = ["timestamp,symbol,side,size,price,tickDirection,trdMatchID,grossValue,homeNotional,foreignNotional,RPI"]
    for seconds, side, size in rows:
        lines.append(f"{EPOCH + seconds:.4f},BTCUSDT,{side},{size},65000.0,PlusTick,x,1,1,1,0")
    return gzip.compress("\n".join(lines).encode())


def _spot_file(rows):
    lines = ["id,timestamp,price,volume,side,rpi"]
    for index, (seconds, side, size) in enumerate(rows):
        lines.append(f"{index},{int((EPOCH + seconds) * 1000)},65000.0,{size},{side},0")
    return gzip.compress("\n".join(lines).encode())


@pytest.fixture(autouse=True)
def runtime_root(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBE_TRADING_HOME", str(tmp_path / "home"))


@pytest.mark.parametrize(("code", "symbol", "market"), [
    ("BTC-USDT-PERP", "BTCUSDT", "perp"), ("eth-usdt", "ETHUSDT", "spot")])
def test_archive_symbol(code, symbol, market):
    assert oa.archive_symbol(code) == (symbol, market)


def test_archive_symbol_refuses_non_crypto_codes():
    with pytest.raises(ValueError):
        oa.archive_symbol("AAPL.US")


def test_archive_urls_follow_both_published_layouts():
    assert oa.archive_url("BTCUSDT", "perp", DAY) == "https://public.bybit.com/trading/BTCUSDT/BTCUSDT2026-11-16.csv.gz"
    assert oa.archive_url("BTCUSDT", "spot", DAY) == "https://public.bybit.com/spot/BTCUSDT/BTCUSDT_2026-11-16.csv.gz"


@pytest.mark.parametrize(("market", "build"), [("perp", _perp_file), ("spot", _spot_file)])
def test_parse_day_aggregates_taker_volume_per_minute(market, build):
    side = ("Buy", "Sell") if market == "perp" else ("buy", "sell")
    raw = build([(1, side[0], 2.0), (30, side[1], 0.5), (61, side[1], 1.5)])
    minutes = oa.parse_day(raw, market)
    assert list(minutes.index) == [pd.Timestamp("2026-11-16 00:00"), pd.Timestamp("2026-11-16 00:01")]
    assert minutes["buy_volume"].tolist() == [2.0, 0.0]
    assert minutes["sell_volume"].tolist() == [0.5, 1.5]
    assert minutes["trade_count"].tolist() == [2, 1]


def test_days_are_downloaded_once_then_read_from_cache():
    calls = []

    def download(url):
        calls.append(url)
        return _perp_file([(5, "Buy", 1.0)])

    first, missing = oa.load_minutes("BTCUSDT", "perp", DAY, DAY, download=download)
    again, _ = oa.load_minutes("BTCUSDT", "perp", DAY, DAY, download=download)
    assert len(calls) == 1 and missing == []
    assert again["buy_volume"].tolist() == first["buy_volume"].tolist() == [1.0]


def test_a_missing_day_is_not_cached_and_reads_as_unavailable():
    calls = []

    def download(url):
        calls.append(url)
        return None

    oa.load_minutes("BTCUSDT", "perp", DAY, DAY, download=download)
    oa.load_minutes("BTCUSDT", "perp", DAY, DAY, download=download)
    assert len(calls) == 2  # retried: Bybit may publish it later


def _bars(start="2026-11-16 00:00", periods=3, freq="5min"):
    index = pd.date_range(start, periods=periods, freq=freq)
    return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=index)


def test_minutes_land_in_the_bar_they_belong_to_and_cvd_runs():
    minutes = pd.DataFrame(
        {"buy_volume": [1.0, 2.0, 0.0, 9.0], "sell_volume": [0.5, 0.0, 4.0, 9.0], "trade_count": [1, 1, 1, 1]},
        index=pd.to_datetime(["2026-11-16 00:00", "2026-11-16 00:04", "2026-11-16 00:05", "2026-11-16 00:15"]),
    )
    out = oa.attach_orderflow(_bars(), minutes, [])
    assert out["buy_volume"].tolist() == [3.0, 0.0, 0.0]  # 00:15 is after the last bar closes
    assert out["delta"].tolist() == [2.5, -4.0, 0.0]
    assert out["cvd"].tolist() == [2.5, -1.5, -1.5]
    assert out["delta_ratio"].iloc[0] == pytest.approx(2.5 / 3.5)
    assert np.isnan(out["delta_ratio"].iloc[2])
    assert out["orderflow_available"].all()


def test_bars_on_a_missing_day_get_nan_not_zero():
    bars = _bars("2026-11-15 23:50", periods=4)
    minutes = pd.DataFrame({"buy_volume": [1.0], "sell_volume": [0.0], "trade_count": [1]},
                           index=pd.to_datetime(["2026-11-15 23:51"]))
    out = oa.attach_orderflow(bars, minutes, [DAY])
    # 23:55 closes exactly at midnight, so only the bars from 00:00 touch the missing day.
    assert out["orderflow_available"].tolist() == [True, True, False, False]
    assert out["delta"].tolist()[:2] == [1.0, 0.0] and out["delta"].iloc[2:].isna().all()
    assert out["cvd"].iloc[2:].isna().all()


def test_enrichment_via_config_uses_the_perp_archive_for_perp_codes():
    seen = []

    def download(url):
        seen.append(url)
        return _perp_file([(10, "Buy", 3.0), (70, "Sell", 1.0)])

    out = oa.enrich_with_orderflow({"BTC-USDT-PERP": _bars(periods=2, freq="1min")},
                                   {"orderflow": {"source": "bybit_archive"}}, download=download)
    frame = out["BTC-USDT-PERP"]
    assert seen == ["https://public.bybit.com/trading/BTCUSDT/BTCUSDT2026-11-16.csv.gz"]
    assert frame["delta"].tolist() == [3.0, -1.0]
    assert set(oa.COLUMNS) <= set(frame.columns)


def test_market_override_reads_perp_flow_for_a_spot_price_code():
    seen = []
    oa.enrich_with_orderflow({"BTC-USDT": _bars(periods=1)}, {"orderflow": {"market": "perp"}},
                             download=lambda url: seen.append(url) or _perp_file([(1, "Buy", 1.0)]))
    assert "/trading/" in seen[0]


def test_disabled_or_unknown_source():
    data = {"BTC-USDT": _bars()}
    assert oa.enrich_with_orderflow(data, {}) is data
    with pytest.raises(ValueError):
        oa.enrich_with_orderflow(data, {"orderflow": {"source": "atas"}})


def test_engine_hook_fails_loudly_when_requested_and_broken(monkeypatch):
    def boom(url):
        raise ConnectionError("blocked")

    monkeypatch.setattr(oa, "_download", boom)
    with pytest.raises(RuntimeError, match="orderflow requested"):
        _maybe_enrich_orderflow({"BTC-USDT": _bars()}, {"orderflow": True})
    data = {"BTC-USDT": _bars()}
    assert _maybe_enrich_orderflow(data, {}) is data


def test_orderflow_is_a_recognized_backtest_config_key():
    assert "orderflow" not in _unrecognized_config_keys({"codes": ["BTC-USDT"], "orderflow": True})
