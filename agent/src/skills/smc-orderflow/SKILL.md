---
name: smc-orderflow
description: "Crypto strategy: Smart Money Concepts entries (BOS/CHoCH, order blocks, FVG retests) confirmed by order flow (delta or absorption from the Bybit trade archive), with a stop-loss and take-profit fixed on every position. Backtestable on USDT perpetuals or spot."
category: strategy
---

# SMC + Order Flow (crypto)

## Purpose

Trade Smart Money Concepts structure only when the tape agrees. Structure says
WHERE a reaction is likely (an order block or fair value gap after a break of
structure); order flow says WHETHER the reaction is happening (aggressive
takers in the trade's direction, or opposite aggression being absorbed).

## Data

Order flow comes from Bybit's public trade archive, attached to the bars by the
backtest itself — no API key:

```json
{
  "codes": ["BTC-USDT-PERP"],
  "source": "ccxt",
  "interval": "15m",
  "start_date": "2026-09-01",
  "end_date": "2026-10-31",
  "leverage": 3,
  "margin_mode": "isolated",
  "orderflow": {"source": "bybit_archive", "market": "perp"}
}
```

- Columns added per bar: `buy_volume`, `sell_volume`, `delta`, `cvd`,
  `delta_ratio`, `trade_count`, `orderflow_available` (base-asset volumes).
- A day of BTCUSDT perpetual trades is ~90 MB to download once; only the
  per-minute aggregate is cached (`~/.vibe-trading/cache/orderflow/`). Keep
  first runs to weeks, not years, and tell the user about the download.
- A day the archive does not have yet (today) has NaN order flow, and the
  engine never confirms an entry on a NaN bar.
- Prices come from the configured `source`; for Bybit prices set
  `CCXT_EXCHANGE=bybit` with `source: "ccxt"`.

## Signal logic (`example_signal_engine.py`)

| Step | Rule |
|---|---|
| Swings | `swing_length`-bar pivots, used only after they are confirmed |
| Structure | Close beyond the last swing high/low: BOS (with trend) or CHoCH (reversal) |
| Zones | Last opposite candle of the breaking move = order block; 3-bar gap in trend = FVG |
| Entry | Trend up and price retests a live bullish zone; mirror for shorts |
| Confirmation | `confirm="either"`: delta in the trade's direction, or absorption (opposite delta ≥ 2× median that failed to move the close) |
| Stop-loss | Just beyond the zone (`stop_buffer_pct`) |
| Take-profit | `rr` × risk (default 2R) |

Parameters: `swing_length=5`, `rr=2.0`, `confirm` (`either`/`delta`/`absorption`/`none`),
`absorption_multiplier=2.0`, `absorption_window=50`, `zone_max_age=60`,
`stop_buffer_pct=0.05`, `allow_short=True`. `engine.trades[code]` lists every
trade with its zone, stop, target and exit reason.

## Risk rules (non-negotiable)

- Every position has a stop-loss and a take-profit from the moment it opens.
- `leverage` at most 5 and `margin_mode: "isolated"` in every config.
- Stops and targets are evaluated on bar highs/lows, and the exit fills at the
  NEXT bar's open — slower than a resting stop or limit order. Report this:
  realised R multiples in the backtest are worse than the planned `rr`, and a
  gap through the stop fills beyond it.
- A backtest is not a forecast. Report the sample size (trades), the period,
  costs and funding, and compare with `confirm="none"` to show what the order
  flow filter actually added.
