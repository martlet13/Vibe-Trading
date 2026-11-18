"""Order-flow analytics: tape aggregation and order-book structure.

Pure functions over trades and book snapshots, with no network access. The
live ``orderflow`` tool and the trade-archive backtest loader share them, so a
delta computed in a chat answer and a delta a strategy trades on are the same
number.

A trade's ``side`` is always the TAKER side, as every venue here reports it: a
buy is an aggressive buyer lifting the ask. Delta is taker-buy volume minus
taker-sell volume, in the base asset.
"""

from src.orderflow.analytics import (
    Trade,
    book_bands,
    build_bars,
    detect_absorption,
    detect_cvd_divergence,
    find_walls,
    nice_step,
    normalize_trades,
    volume_profile,
)

__all__ = [
    "Trade",
    "book_bands",
    "build_bars",
    "detect_absorption",
    "detect_cvd_divergence",
    "find_walls",
    "nice_step",
    "normalize_trades",
    "volume_profile",
]
