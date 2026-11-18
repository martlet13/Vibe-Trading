---
name: order-flow
description: "Crypto order-flow reading with the orderflow tool (Bybit/Binance, spot or USDT perpetual): book imbalance and walls, delta, CVD, footprint imbalances, volume profile (POC / value area), absorption and CVD divergence — the readings platforms like ATAS show, from public exchange data."
category: analysis
---

# Order Flow (crypto)

## When to use

The user asks about the order book, "стакан", ATAS-style footprint, delta, CVD,
volume profile, absorption, who is aggressive right now, or wants order flow to
confirm a Smart Money Concepts setup (order block, FVG, liquidity sweep).

## How to call

```json
{"symbol": "BTC-USDT", "exchange": "bybit", "market": "perp", "bar_minutes": 1, "collect_seconds": 60}
```

- Bybit USDT perpetual is the default venue. Use `market: "spot"` only when the
  user asks for spot — Bybit spot REST returns just 60 trades.
- `collect_seconds: 0` is one REST call: fast, but the tape holds only the most
  recent trades. For any statement about "the last N minutes", stream with
  `collect_seconds` (up to 180) and check `coverage.seconds_covered`.
- Always state `coverage` (source, trades used, time span) in the answer.

## Reading the output

| Field | Meaning |
|---|---|
| `book.bands[].imbalance` | (bid − ask) / (bid + ask) notional near mid; > 0 bid-heavy |
| `book.walls[]` | Large resting levels; `persistence` < 1 means the level came and went (possible spoof) |
| `flow.delta`, `bars[].delta` | Taker buys − taker sells, in the base asset |
| `bars[].cvd` | Running delta from the first bar of the window (relative, not absolute) |
| `bars[].stacked_*_imbalances` | Footprint levels where one side outweighed the diagonal 3:1 |
| `profile.poc_price`, `value_area_*` | Price with most volume; range holding 70% of volume |
| `signals.absorption` | Heavy one-sided flow that did not move price (sell_absorption = bullish read) |
| `signals.cvd_divergence` | Price at a new extreme while CVD is not |

## Discipline

- Order flow describes the recent past. Never present it as a forecast or a
  trade instruction; a wall can be pulled at any moment.
- Quote numbers exactly as returned, with their units: volumes and delta are in
  the base asset, prices in the quote asset (USDT for BTC-USDT).
- Combining with SMC: an order-block or FVG retest is stronger when the bar
  touching it shows absorption against the move or delta turning in the trade's
  direction. Any order must still go through the trading tools and the user's
  mandate (stop-loss and take-profit on every position, isolated margin, ≤ 5x).
