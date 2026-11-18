"""Smart Money Concepts entries confirmed by order flow, with stop-loss and take-profit.

Structure (pure pandas, no extra dependency):

* A swing high/low at bar ``j`` is a ``swing_length``-bar pivot. It is only
  CONFIRMED at bar ``j + swing_length`` and used from then on — never earlier.
* A close beyond the last confirmed swing high is a bullish break: a BOS when
  the trend was already up, a CHoCH when it was down. It sets the trend and
  marks the last bearish candle of the move as a bullish order block (OB).
  Bearish breaks mirror this.
* A bullish fair value gap (FVG) is ``low[i] > high[i-2]`` in an up-trend; the
  gap ``[high[i-2], low[i]]`` becomes a zone. Bearish FVGs mirror it.

Entry (long; short mirrors): trend up, price trades back into a live bullish OB
or FVG (``low <= zone top`` and ``close > zone bottom``), and the order flow on
that bar confirms — ``confirm="delta"``: delta > 0; ``"absorption"``: heavy
taker selling (|delta| ≥ ``absorption_multiplier`` × the median |delta| of the
previous ``absorption_window`` bars) that failed to push the close below the
open; ``"either"`` (default): one of the two; ``"none"``: no confirmation.

Every position carries a stop-loss and a take-profit fixed at entry: the stop
just beyond the zone (``stop_buffer_pct``), the target at ``rr`` × risk. The
engine returns 0 on the bar that touches either level; the backtest executes
that at the next bar's open, so a gap through the stop fills worse than the
stop — a conservative reading. If one bar touches both, the stop wins.

Requires the order-flow columns the backtest attaches with
``{"orderflow": {"source": "bybit_archive"}}``. Run it with
``"leverage" <= 5`` and ``"margin_mode": "isolated"``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_CONFIRM_MODES = ("either", "delta", "absorption", "none")


class _Zone:
    """A price zone a retest may enter from: an order block or a fair value gap."""

    __slots__ = ("direction", "kind", "top", "bottom", "created")

    def __init__(self, direction: int, kind: str, top: float, bottom: float, created: int) -> None:
        self.direction = direction  # 1 = bullish (long), -1 = bearish (short)
        self.kind = kind  # "ob" or "fvg"
        self.top = top
        self.bottom = bottom
        self.created = created


class SignalEngine:
    """SMC structure + order-flow confirmation, one position at a time per symbol.

    Attributes:
        trades: After ``generate``, one dict per closed or open trade with its
            entry bar, side, zone, stop, target and exit reason.
    """

    def __init__(
        self,
        swing_length: int = 5,
        rr: float = 2.0,
        confirm: str = "either",
        absorption_multiplier: float = 2.0,
        absorption_window: int = 50,
        zone_max_age: int = 60,
        stop_buffer_pct: float = 0.05,
        allow_short: bool = True,
    ) -> None:
        if confirm not in _CONFIRM_MODES:
            raise ValueError(f"confirm must be one of {_CONFIRM_MODES}")
        if swing_length < 1 or rr <= 0 or absorption_window < 3 or zone_max_age < 1 or stop_buffer_pct < 0:
            raise ValueError("invalid SMC order-flow parameters")
        self.swing_length = swing_length
        self.rr = rr
        self.confirm = confirm
        self.absorption_multiplier = absorption_multiplier
        self.absorption_window = absorption_window
        self.zone_max_age = zone_max_age
        self.stop_buffer_pct = stop_buffer_pct
        self.allow_short = allow_short
        self.trades: Dict[str, List[dict]] = {}

    def generate(self, data_map: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """Return a target direction per bar (1 long, -1 short, 0 flat) for each symbol."""
        result: Dict[str, pd.Series] = {}
        for code, frame in data_map.items():
            if self.confirm != "none" and "delta" not in frame.columns:
                raise ValueError(
                    f"{code} has no order-flow columns; enable them with "
                    '{"orderflow": {"source": "bybit_archive"}} in the backtest config'
                )
            signal, trades = self._run(frame)
            result[code] = signal
            self.trades[code] = trades
        return result

    def _confirmed(self, direction: int, i: int, open_: np.ndarray, close: np.ndarray,
                   delta: np.ndarray, typical: np.ndarray) -> bool:
        if self.confirm == "none":
            return True
        if np.isnan(delta[i]):
            return False
        with_flow = delta[i] * direction > 0
        absorbed = (
            not np.isnan(typical[i])
            and typical[i] > 0
            and delta[i] * direction < 0
            and abs(delta[i]) >= self.absorption_multiplier * typical[i]
            and (close[i] - open_[i]) * direction >= 0
        )
        if self.confirm == "delta":
            return with_flow
        if self.confirm == "absorption":
            return absorbed
        return with_flow or absorbed

    def _run(self, frame: pd.DataFrame) -> tuple[pd.Series, List[dict]]:
        open_ = frame["open"].to_numpy(dtype=float)
        high = frame["high"].to_numpy(dtype=float)
        low = frame["low"].to_numpy(dtype=float)
        close = frame["close"].to_numpy(dtype=float)
        n = len(frame)
        if "delta" in frame.columns:
            delta = frame["delta"].to_numpy(dtype=float)
            # Median |delta| of the PREVIOUS window only: shift(1) keeps bar i out.
            typical = (
                frame["delta"].abs().rolling(self.absorption_window, min_periods=self.absorption_window)
                .median().shift(1).to_numpy(dtype=float)
            )
        else:
            delta = np.full(n, np.nan)
            typical = np.full(n, np.nan)

        L = self.swing_length
        signal = np.zeros(n)
        trades: List[dict] = []
        trend = 0
        swing_high: Optional[tuple[float, int]] = None
        swing_low: Optional[tuple[float, int]] = None
        zones: List[_Zone] = []
        position = 0
        stop = target = 0.0
        current: Optional[dict] = None

        for i in range(n):
            exited = False
            # 1. Pivots confirmed at this bar (centre j = i - L, window [j-L, j+L]).
            j = i - L
            if j - L >= 0:
                if high[j] == high[j - L: i + 1].max():
                    swing_high = (high[j], j)
                if low[j] == low[j - L: i + 1].min():
                    swing_low = (low[j], j)

            # 2. Exits on the levels fixed at entry; the entry bar itself is excluded.
            if position and current is not None and i > current["signal_bar"]:
                hit_stop = low[i] <= stop if position > 0 else high[i] >= stop
                hit_target = high[i] >= target if position > 0 else low[i] <= target
                if hit_stop or hit_target:
                    current.update(exit_bar=str(frame.index[i]), exit_reason="stop_loss" if hit_stop else "take_profit")
                    position, current, exited = 0, None, True

            # 3. Structure breaks by close; each swing breaks once.
            if swing_high is not None and close[i] > swing_high[0]:
                trend = 1  # a BOS when already up, a CHoCH when it was down
                zones = [z for z in zones if z.direction == 1]
                start = swing_low[1] if swing_low is not None and swing_low[1] < i else max(0, i - 20)
                bearish = [k for k in range(start, i) if close[k] < open_[k]]
                if bearish:
                    k = bearish[-1]
                    zones.append(_Zone(1, "ob", high[k], low[k], i))
                swing_high = None
            elif swing_low is not None and close[i] < swing_low[0]:
                trend = -1
                zones = [z for z in zones if z.direction == -1]
                start = swing_high[1] if swing_high is not None and swing_high[1] < i else max(0, i - 20)
                bullish = [k for k in range(start, i) if close[k] > open_[k]]
                if bullish:
                    k = bullish[-1]
                    zones.append(_Zone(-1, "ob", high[k], low[k], i))
                swing_low = None

            # 4. Fair value gaps in the trend's direction.
            if i >= 2:
                if trend == 1 and low[i] > high[i - 2]:
                    zones.append(_Zone(1, "fvg", low[i], high[i - 2], i))
                elif trend == -1 and high[i] < low[i - 2]:
                    zones.append(_Zone(-1, "fvg", low[i - 2], high[i], i))

            # 5. Entry on a retest of a zone created on an earlier bar.
            # The exit order fills at the next open, so no new entry on the exit bar.
            if position == 0 and not exited and trend != 0 and (trend == 1 or self.allow_short):
                for zone in reversed(zones):
                    if zone.direction != trend or zone.created >= i or i - zone.created > self.zone_max_age:
                        continue
                    touched = (
                        low[i] <= zone.top and close[i] > zone.bottom if trend == 1
                        else high[i] >= zone.bottom and close[i] < zone.top
                    )
                    if not touched or not self._confirmed(trend, i, open_, close, delta, typical):
                        continue
                    if trend == 1:
                        stop = zone.bottom * (1 - self.stop_buffer_pct / 100)
                        risk = close[i] - stop
                        target = close[i] + self.rr * risk
                    else:
                        stop = zone.top * (1 + self.stop_buffer_pct / 100)
                        risk = stop - close[i]
                        target = close[i] - self.rr * risk
                    if risk <= 0:
                        continue
                    position = trend
                    current = {
                        "signal_bar": i, "time": str(frame.index[i]), "side": "long" if trend == 1 else "short",
                        "zone": zone.kind, "entry_ref": float(close[i]), "stop_loss": float(stop), "take_profit": float(target),
                        "delta": None if np.isnan(delta[i]) else float(delta[i]),
                    }
                    trades.append(current)
                    zones.remove(zone)
                    break

            # 6. A close through a zone kills it; old zones expire.
            zones = [
                z for z in zones
                if i - z.created <= self.zone_max_age
                and not (z.direction == 1 and close[i] < z.bottom)
                and not (z.direction == -1 and close[i] > z.top)
            ]
            signal[i] = position

        for trade in trades:
            trade.pop("signal_bar", None)
        return pd.Series(signal, index=frame.index), trades
