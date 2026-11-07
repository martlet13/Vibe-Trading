"""Bybit ccxt connector with host-separated paper and live profiles.

Every client is pinned to one category (spot, or linear perpetuals for a
futures profile) and checked against the host its profile declares: a paper
profile must resolve every ccxt endpoint to ``api-testnet.bybit.com`` and a
live profile to ``api.<hostname>``. The check reads the client's own resolved
URLs, so a sandbox flag that failed to apply fails closed instead of sending a
testnet intent to the live host.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping
from urllib.parse import urlparse
from urllib.request import getproxies

from src.config.paths import get_runtime_root
from src.trading.connectors.binance.shaping import (
    as_iter as _as_iter,
    nonzero_balances as _nonzero_balances,
    obj_get as _obj_get,
    ohlcv_to_dict as _ohlcv_to_dict,
    order_to_dict as _order_to_dict,
    to_float as _to_float,
    trade_to_dict as _trade_to_dict,
)

CONFIG_FILENAME = "bybit.json"

#: Profiles this connector understands and their default account environment.
PROFILE_ENVIRONMENTS = {
    "paper": "paper",
    "live-readonly": "live",
    "live": "live",
}

#: Bybit's testnet exists only under bybit.com.
TESTNET_HOST = "api-testnet.bybit.com"
DEFAULT_HOSTNAME = "bybit.com"
#: Regional mainnet domains Bybit serves the V5 API from. Each region has its
#: own accounts and keys; anything else is rejected so a typo can never route
#: a signed request to an unexpected host.
LIVE_HOSTNAMES = frozenset({"bybit.com", "bytick.com", "bybit.eu", "bybit.nl", "bybit.com.hk", "bybit.kz"})

_QUOTE_ASSETS = ("USDT", "USDC", "USDE", "DAI", "BTC", "ETH", "EUR", "BRL", "TRY")

#: USD stablecoins held as cash. Their balances carry no market exposure.
_CASH_ASSETS = frozenset({"USDT", "USDC", "USDE", "DAI", "FDUSD", "TUSD", "BUSD"})


class BybitDependencyError(RuntimeError):
    """Raised when the optional ``ccxt`` package is not installed."""


class BybitConfigError(RuntimeError):
    """Raised when the connector configuration is missing or invalid."""


@dataclass(frozen=True)
class BybitConfig:
    """Bybit connector connection settings."""

    api_key: str = ""
    api_secret: str = ""
    profile: str = "paper"
    hostname: str = DEFAULT_HOSTNAME
    timeout: float = 15.0
    readonly: bool = True
    futures: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> "BybitConfig":
        """Build a config from a JSON-like mapping, normalizing the profile."""
        payload = dict(data or {})
        profile = str(payload.get("profile") or "paper").strip().lower()
        if profile not in PROFILE_ENVIRONMENTS:
            raise BybitConfigError("profile must be 'paper', 'live-readonly' or 'live'")
        hostname = str(payload.get("hostname") or DEFAULT_HOSTNAME).strip().lower()
        if hostname not in LIVE_HOSTNAMES:
            raise BybitConfigError(
                "hostname must be one of: " + ", ".join(sorted(LIVE_HOSTNAMES))
            )
        if PROFILE_ENVIRONMENTS[profile] == "paper" and hostname != DEFAULT_HOSTNAME:
            raise BybitConfigError("the Bybit testnet is served only from bybit.com")
        raw_timeout = payload.get("timeout")
        try:
            timeout = float(15.0 if raw_timeout in (None, "") else raw_timeout)
        except (TypeError, ValueError) as exc:
            raise BybitConfigError("timeout must be a positive number") from exc
        if not (math.isfinite(timeout) and timeout > 0):
            raise BybitConfigError("timeout must be a positive number")
        futures = payload.get("futures", False)
        if not isinstance(futures, bool):
            raise BybitConfigError("futures must be true or false")
        if futures and profile == "live-readonly":
            raise BybitConfigError("futures trading is available only through the futures trade profiles")
        return cls(
            api_key=str(payload.get("api_key") or "").strip(),
            api_secret=str(payload.get("api_secret") or "").strip(),
            profile=profile,
            hostname=hostname,
            timeout=timeout,
            readonly=bool(payload.get("readonly", True)),
            futures=futures,
        )

    def with_overrides(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        profile: str | None = None,
        hostname: str | None = None,
    ) -> "BybitConfig":
        """Return a copy with CLI/tool overrides applied."""
        payload = asdict(self)
        if api_key is not None:
            payload["api_key"] = api_key
        if api_secret is not None:
            payload["api_secret"] = api_secret
        if profile is not None:
            payload["profile"] = profile
        if hostname is not None:
            payload["hostname"] = hostname
        return BybitConfig.from_mapping(payload)

    @property
    def environment(self) -> str:
        """Return ``paper`` or ``live`` for this profile."""
        return PROFILE_ENVIRONMENTS.get(self.profile, "paper")

    @property
    def is_testnet(self) -> bool:
        """Return whether this profile targets the testnet host/key."""
        return self.environment == "paper"

    @property
    def host(self) -> str:
        """Return the REST host this profile connects to."""
        return TESTNET_HOST if self.is_testnet else f"api.{self.hostname}"


_OVERRIDE_KEYS = ("api_key", "api_secret", "profile", "hostname")


def build_config(
    profile_config: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> BybitConfig:
    """Resolve the effective config: saved file ← profile defaults ← CLI overrides.

    Credentials (``api_key`` / ``api_secret``) come from the saved
    ``~/.vibe-trading/bybit.json``; the selected connector profile supplies the
    ``profile`` intent; CLI/tool overrides win last.
    """
    base = asdict(load_config())
    for key, value in dict(profile_config or {}).items():
        if value is not None:
            base[key] = value
    # Only the selected profile decides whether this is a futures connection; a
    # saved file never turns a spot profile into a derivatives one.
    base["futures"] = bool(dict(profile_config or {}).get("futures", False))
    cfg = BybitConfig.from_mapping(base)
    clean = {k: v for k, v in dict(overrides or {}).items() if k in _OVERRIDE_KEYS and v not in (None, "")}
    return cfg.with_overrides(**clean) if clean else cfg


def config_path() -> Path:
    """Return the user-level Bybit config path."""
    return get_runtime_root() / CONFIG_FILENAME


def load_config() -> BybitConfig:
    """Load Bybit settings from ``~/.vibe-trading/bybit.json``."""
    path = config_path()
    if not path.exists():
        return BybitConfig()
    try:
        return BybitConfig.from_mapping(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, json.JSONDecodeError, BybitConfigError) as exc:
        raise BybitConfigError(f"invalid Bybit config at {path}: {exc}") from exc


def save_config(config: BybitConfig) -> Path:
    """Persist Bybit settings with owner-only permissions."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def normalize_symbol(symbol: str) -> str:
    """Normalize a symbol to ccxt unified ``BASE/QUOTE`` spot format."""
    clean = (symbol or "").strip().upper().replace("-", "/")
    if "/" in clean:
        return clean
    for quote in _QUOTE_ASSETS:
        if clean.endswith(quote) and len(clean) > len(quote):
            return f"{clean[: -len(quote)]}/{quote}"
    return clean


def ccxt_available() -> bool:
    """Return whether the optional ``ccxt`` library can be imported."""
    try:
        _require_ccxt()
        return True
    except BybitDependencyError:
        return False


def check_status(config: BybitConfig | None = None) -> dict[str, Any]:
    """Check SDK readiness, config completeness, and host separation.

    Returns a JSON-serializable health report. Does not place or mutate any
    broker state.
    """
    cfg = config or load_config()
    report: dict[str, Any] = {
        "status": "ok",
        "config": _public_config(cfg),
        "sdk": {"package": "ccxt", "installed": ccxt_available()},
        "paper_guard": "host_separated",
        "host": cfg.host,
    }

    missing = _missing_fields(cfg)
    if missing:
        report["status"] = "error"
        report["error"] = f"Bybit connector not configured: missing {', '.join(missing)}."
        return report

    if not report["sdk"]["installed"]:
        report["status"] = "error"
        report["error"] = "Optional dependency missing: install with `pip install ccxt`."
        return report

    try:
        snapshot = get_account_snapshot(cfg)
    except Exception as exc:  # noqa: BLE001 - health endpoint reports cleanly
        report["status"] = "error"
        report["error"] = str(exc)
        if _is_region_block(exc):
            # ccxt maps CloudFront's 403 to RateLimitExceeded; say what it is.
            report["error_code"] = "bybit_restricted_location"
            report["error"] = (
                f"Bybit refused the connection from this network location ({cfg.host} returned "
                "HTTP 403: CloudFront blocks the country). Connect from a region Bybit serves, "
                "or set hostname to the regional site your account belongs to."
            )
        return report

    report["account"] = {
        "profile": cfg.profile,
        "is_testnet": cfg.is_testnet,
        "balances": len(snapshot.get("balances", [])),
    }
    return report


def get_account_snapshot(config: BybitConfig | None = None) -> dict[str, Any]:
    """Fetch the non-zero spot balances (``free`` / ``used`` / ``total``)."""
    cfg = config or load_config()
    ex = _exchange(cfg)
    if cfg.futures:
        return _futures().get_account_snapshot(cfg, ex)
    balance = ex.fetch_balance()
    return {
        "status": "ok",
        "profile": cfg.profile,
        "is_testnet": cfg.is_testnet,
        "host": cfg.host,
        "paper_guard": "host_separated",
        "balances": _nonzero_balances(balance),
    }


def get_positions(config: BybitConfig | None = None) -> dict[str, Any]:
    """Fetch spot holdings, shaped from the non-zero balances.

    Bybit spot has no positions; each non-zero asset balance is one row. Each
    row also carries ``value_usd``, the exposure the live mandate gate sums:
    a crypto balance is valued at its ``<ASSET>/USDT`` last price, and a
    stablecoin balance is cash, not exposure, so it counts as zero. A balance
    with no USDT price gets no ``value_usd``, which the gate reads as
    unpriceable and refuses the order (fail-closed).
    """
    cfg = config or load_config()
    ex = _exchange(cfg)
    if cfg.futures:
        return _futures().get_positions(cfg, ex)
    balance = ex.fetch_balance()
    rows = [
        {
            "symbol": _obj_get(row, "asset"),
            "quantity": _obj_get(row, "total"),
            "free": _obj_get(row, "free"),
            "used": _obj_get(row, "used"),
            "source": "spot",
        }
        for row in _nonzero_balances(balance)
    ]
    result = {
        "status": "ok",
        "profile": cfg.profile,
        "is_testnet": cfg.is_testnet,
        "paper_guard": "host_separated",
        "positions": rows,
    }
    note = _attach_exposure_values(cfg, rows)
    if note:
        result["note"] = note
    return result


def _attach_exposure_values(cfg: BybitConfig, rows: list[dict[str, Any]]) -> str | None:
    """Set ``value_usd`` on each row from one public ticker read; return a note on gaps."""
    to_price: dict[str, dict[str, Any]] = {}
    for row in rows:
        asset = str(row.get("symbol") or "").upper()
        if asset in _CASH_ASSETS:
            row["value_usd"] = 0.0
        elif asset:
            to_price[f"{asset}/USDT"] = row
    if not to_price:
        return None
    try:
        tickers = _public_exchange(cfg).fetch_tickers(list(to_price))
    except Exception as exc:  # noqa: BLE001 - balances stay usable; the gate fails closed on the gap
        return f"USDT prices unavailable ({type(exc).__name__}); holdings are left unvalued."
    unpriced = []
    for pair, row in to_price.items():
        last = _to_float(_obj_get(_obj_get(tickers, pair, {}), "last"))
        quantity = _to_float(row.get("quantity"))
        if _positive_finite(last) and quantity is not None and math.isfinite(quantity):
            row["value_usd"] = abs(quantity) * last
        else:
            unpriced.append(pair.split("/", 1)[0])
    return f"No USDT price for: {', '.join(sorted(unpriced))}." if unpriced else None


def get_open_orders(config: BybitConfig | None = None, *, include_executions: bool = False) -> dict[str, Any]:
    """Fetch open spot orders and, optionally, recent personal trades.

    Only the "this call needs a symbol" family degrades to a note; auth,
    network and rate-limit errors propagate so the caller sees a real failure
    instead of a misleading status:ok with an empty list.
    """
    cfg = config or load_config()
    ex = _exchange(cfg)
    if cfg.futures:
        return _futures().get_open_orders(cfg, ex)
    symbol_required = _symbol_required_errors()
    result: dict[str, Any] = {
        "status": "ok",
        "profile": cfg.profile,
        "is_testnet": cfg.is_testnet,
        "paper_guard": "host_separated",
    }
    try:
        open_orders = ex.fetch_open_orders()
        result["open_orders"] = [_order_to_dict(item) for item in _as_iter(open_orders)]
    except symbol_required as exc:
        result["open_orders"] = []
        result["open_orders_note"] = f"fetch_open_orders without a symbol failed: {exc}"
    if include_executions:
        try:
            trades = ex.fetch_my_trades()
            result["executions"] = [_trade_to_dict(item) for item in _as_iter(trades)]
        except symbol_required as exc:
            result["executions"] = []
            result["executions_note"] = f"fetch_my_trades without a symbol failed: {exc}"
    return result


def get_quote(symbol: str, *, config: BybitConfig | None = None, **_: Any) -> dict[str, Any]:
    """Fetch a latest spot ticker snapshot for ``symbol``."""
    cfg = config or load_config()
    ex = _public_exchange(cfg)
    if cfg.futures:
        return _futures().get_quote(cfg, ex, symbol)
    clean = normalize_symbol(symbol)
    ticker = ex.fetch_ticker(clean)
    return {
        "status": "ok",
        "symbol": clean,
        "quote": {
            "bid": _obj_get(ticker, "bid"),
            "ask": _obj_get(ticker, "ask"),
            "last": _obj_get(ticker, "last"),
            "high": _obj_get(ticker, "high"),
            "low": _obj_get(ticker, "low"),
            "volume": _obj_get(ticker, "baseVolume"),
            "time": str(_obj_get(ticker, "timestamp", "")),
        },
    }


#: Project/canonical period token → ccxt unified timeframe (lowercase).
_TIMEFRAME_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "1H": "1h", "4h": "4h", "4H": "4h",
    "1d": "1d", "1D": "1d", "1w": "1w", "1W": "1w", "1M": "1M",
}


def get_historical_bars(
    symbol: str,
    *,
    config: BybitConfig | None = None,
    period: str = "1d",
    limit: int = 90,
    **_: Any,
) -> dict[str, Any]:
    """Fetch historical spot OHLCV bars for ``symbol``."""
    cfg = config or load_config()
    ex = _public_exchange(cfg)
    timeframe = _TIMEFRAME_MAP.get(period.strip(), "1d")
    if cfg.futures:
        return _futures().get_historical_bars(cfg, ex, symbol, timeframe, limit, period)
    clean = normalize_symbol(symbol)
    bars = ex.fetch_ohlcv(clean, timeframe=timeframe, limit=int(limit))
    return {
        "status": "ok",
        "symbol": clean,
        "period": period,
        "bars": [_ohlcv_to_dict(item) for item in _as_iter(bars)],
    }


# ---------------------------------------------------------------------------
# Order placement (write path; guarded by host separation + profile readonly)
# ---------------------------------------------------------------------------

#: ccxt ``timeInForce`` values Bybit spot accepts for limit orders. Bybit has
#: no DAY policy; the unified ``"day"`` intent maps to GTC, the Bybit default.
_TIME_IN_FORCE_MAP = {
    "day": "GTC",
    "gtc": "GTC",
    "ioc": "IOC",
    "fok": "FOK",
}


def place_order(
    config: BybitConfig | None = None,
    *,
    symbol: str,
    side: str,
    quantity: float | None = None,
    notional: float | None = None,
    order_type: str = "market",
    limit_price: float | None = None,
    time_in_force: str = "day",
    stop_loss: float | None = None,
    take_profit: float | None = None,
    leverage: float | None = None,
    reduce_only: bool = False,
) -> dict[str, Any]:
    """Place a spot order on Bybit via ccxt's unified ``create_order``.

    A futures profile routes to :func:`src.trading.connectors.bybit.futures.place_order`
    instead, which takes ``stop_loss``, ``take_profit``, ``leverage`` and
    ``reduce_only``; a spot profile refuses those arguments rather than drop
    the protection a caller asked for.

    The connector executes the intent the caller has already authorized;
    mandate/limit enforcement lives in a higher layer. The client's resolved
    host is checked against the profile before anything is submitted.

    Either ``quantity`` (base-asset amount) or ``notional`` (quote-asset
    amount) must be given, never both. ``notional`` is only supported for
    market orders and is sent as ccxt's ``cost`` param, which Bybit sizes in
    the quote coin on both sides. Limit orders require ``quantity`` and
    ``limit_price``.

    Returns:
        On success: ``{"status": "ok", "order_id": str, "symbol", "side",
        "profile", "order_type", "order_status", "filled", "amount",
        "price"}``. On any validation or execution failure: ``{"status":
        "error", "error": str}`` (fail-closed; nothing is submitted on a
        validation error).
    """
    cfg = config or load_config()

    side_clean = str(side or "").strip().lower()
    if side_clean not in ("buy", "sell"):
        return {"status": "error", "error": "side must be 'buy' or 'sell'."}

    type_clean = str(order_type or "").strip().lower()
    if type_clean not in ("market", "limit"):
        return {"status": "error", "error": "order_type must be 'market' or 'limit'."}

    qty_given = quantity is not None
    notional_given = notional is not None
    if qty_given == notional_given:
        return {"status": "error", "error": "provide exactly one of 'quantity' or 'notional'."}

    qty_value = _to_float(quantity) if qty_given else None
    notional_value = _to_float(notional) if notional_given else None
    if qty_given and not _positive_finite(qty_value):
        return {"status": "error", "error": "quantity must be a positive number."}
    if notional_given and not _positive_finite(notional_value):
        return {"status": "error", "error": "notional must be a positive number."}

    if cfg.futures:
        return _place_futures_order(
            cfg, symbol=symbol, side=side_clean, quantity=qty_value, notional_given=notional_given,
            order_type=type_clean, limit_price=limit_price, time_in_force=time_in_force,
            stop_loss=stop_loss, take_profit=take_profit, leverage=leverage, reduce_only=reduce_only,
        )
    if stop_loss is not None or take_profit is not None or leverage is not None or reduce_only:
        return {
            "status": "error",
            "error": "stop_loss, take_profit, leverage and reduce_only apply to futures profiles only.",
        }

    params: dict[str, Any] = {}
    price: float | None = None
    if type_clean == "limit":
        if notional_given:
            return {"status": "error", "error": "limit orders require 'quantity', not 'notional'."}
        price = _to_float(limit_price)
        if not _positive_finite(price):
            return {"status": "error", "error": "limit orders require a positive 'limit_price'."}
        tif = _TIME_IN_FORCE_MAP.get(str(time_in_force or "").strip().lower())
        if tif is None:
            return {"status": "error", "error": "time_in_force must be one of 'day', 'gtc', 'ioc', 'fok'."}
        params["timeInForce"] = tif
        amount = qty_value
    elif notional_given:
        params["cost"] = notional_value
        amount = notional_value
    else:
        amount = qty_value

    clean_symbol = normalize_symbol(symbol)
    if not clean_symbol or "/" not in clean_symbol:
        return {"status": "error", "error": f"could not resolve a valid trading pair from symbol '{symbol}'."}

    try:
        ex = _exchange(cfg)
        order = ex.create_order(clean_symbol, type_clean, side_clean, amount, price, params)
    except Exception as exc:  # noqa: BLE001 - surface any ccxt/auth/network/host error as fail-closed
        return {"status": "error", "error": str(exc)}

    return {
        "status": "ok",
        "order_id": str(_obj_get(order, "id", "")),
        "symbol": _obj_get(order, "symbol", clean_symbol),
        "side": str(_obj_get(order, "side", side_clean)),
        "profile": cfg.profile,
        "is_testnet": cfg.is_testnet,
        "paper_guard": "host_separated",
        "order_type": str(_obj_get(order, "type", type_clean)),
        "order_status": str(_obj_get(order, "status", "")),
        "filled": _obj_get(order, "filled"),
        "amount": _obj_get(order, "amount"),
        "price": _obj_get(order, "price"),
    }


def cancel_order(
    config: BybitConfig | None = None,
    order_id: str = "",
    *,
    symbol: str | None = None,
) -> dict[str, Any]:
    """Cancel an open spot order by id. Bybit REQUIRES the order's symbol.

    Returns:
        On success: ``{"status": "ok", "order_id", "symbol", "side", "profile",
        "order_status"}``. On any validation or execution failure:
        ``{"status": "error", "error": str}`` (fail-closed).
    """
    cfg = config or load_config()

    order_id_clean = str(order_id or "").strip()
    if not order_id_clean:
        return {"status": "error", "error": "order_id is required to cancel an order."}
    if symbol is None or not str(symbol).strip():
        return {"status": "error", "error": "Bybit cancel requires symbol."}

    if cfg.futures:
        try:
            ex = _exchange(cfg)
        except BybitConfigError as exc:
            return {"status": "error", "error": str(exc)}
        return _futures().cancel_order(cfg, ex, order_id_clean, str(symbol))

    clean_symbol = normalize_symbol(symbol)
    if not clean_symbol or "/" not in clean_symbol:
        return {"status": "error", "error": f"could not resolve a valid trading pair from symbol '{symbol}'."}

    try:
        ex = _exchange(cfg)
        order = ex.cancel_order(order_id_clean, clean_symbol)
    except Exception as exc:  # noqa: BLE001 - surface any ccxt/auth/network/host error as fail-closed
        return {"status": "error", "error": str(exc)}

    return {
        "status": "ok",
        "order_id": str(_obj_get(order, "id", order_id_clean)),
        "symbol": _obj_get(order, "symbol", clean_symbol),
        "side": str(_obj_get(order, "side", "")),
        "profile": cfg.profile,
        "is_testnet": cfg.is_testnet,
        "paper_guard": "host_separated",
        "order_status": str(_obj_get(order, "status", "")),
    }


def _place_futures_order(
    cfg: BybitConfig,
    *,
    symbol: str,
    side: str,
    quantity: float | None,
    notional_given: bool,
    order_type: str,
    limit_price: Any,
    time_in_force: str,
    stop_loss: Any,
    take_profit: Any,
    leverage: Any,
    reduce_only: Any,
) -> dict[str, Any]:
    """Check size, price and policy, then route a futures order to the futures module."""
    if notional_given:
        return {"status": "error", "error": "futures orders take 'quantity' in base units, not 'notional'."}
    price: float | None = None
    tif: str | None = None
    if order_type == "limit":
        price = _to_float(limit_price)
        if not _positive_finite(price):
            return {"status": "error", "error": "limit orders require a positive 'limit_price'."}
        tif = _TIME_IN_FORCE_MAP.get(str(time_in_force or "").strip().lower())
        if tif is None:
            return {"status": "error", "error": "time_in_force must be one of 'day', 'gtc', 'ioc', 'fok'."}
    try:
        ex = _exchange(cfg)
    except BybitConfigError as exc:
        return {"status": "error", "error": str(exc)}
    return _futures().place_order(
        cfg, ex, symbol=symbol, side=side, quantity=quantity, order_type=order_type,
        limit_price=price, time_in_force=tif,
        stop_loss=stop_loss, take_profit=take_profit, leverage=leverage, reduce_only=reduce_only,
    )


# ---------------------------------------------------------------------------
# SDK plumbing
# ---------------------------------------------------------------------------


def _futures() -> ModuleType:
    """Import the futures module lazily; it imports helpers from this one."""
    from src.trading.connectors.bybit import futures

    return futures


def _require_ccxt() -> ModuleType:
    try:
        import ccxt  # type: ignore
    except ModuleNotFoundError as exc:
        raise BybitDependencyError("ccxt is not installed; run `pip install ccxt`.") from exc
    return ccxt


def _symbol_required_errors() -> tuple[type[BaseException], ...]:
    """ccxt exception classes that mean "this call needs a symbol" (degrade-only)."""
    ccxt = _require_ccxt()
    names = ("ArgumentsRequired", "BadSymbol", "NotSupported")
    classes = tuple(getattr(ccxt, n) for n in names if hasattr(ccxt, n))
    return classes or (ValueError,)


def _is_region_block(exc: BaseException) -> bool:
    """Return whether an error is CloudFront's country block on Bybit's API."""
    return "block access from your country" in str(exc)


def _positive_finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0


def _public_exchange(cfg: BybitConfig):
    """Use the selected Bybit host without credentials for public market data."""
    return _exchange(replace(cfg, api_key="", api_secret=""))


def _exchange(cfg: BybitConfig):
    """Build a ccxt Bybit spot client and verify it resolves to the profile's host."""
    ccxt = _require_ccxt()
    client_config: dict[str, Any] = {
        "apiKey": cfg.api_key,
        "secret": cfg.api_secret,
        "enableRateLimit": True,
        "timeout": int(cfg.timeout * 1000),
        "hostname": cfg.hostname,
        "options": {
            # ccxt's Bybit default category is swap; spot profiles pin spot and
            # futures profiles pin USDT-margined linear perpetuals.
            "defaultType": "swap" if cfg.futures else "spot",
            "defaultSubType": "linear",
            # Signed Bybit requests carry a timestamp checked against
            # recv_window; measure the exchange clock first so a drifting
            # system clock does not reject otherwise valid keys.
            "adjustForTimeDifference": True,
            "recvWindow": 10_000,
        },
    }
    # ``requests``/ccxt does not consistently inherit the macOS System Proxy;
    # urllib resolves both proxy environment variables and the OS proxy.
    system_proxies = getproxies()
    proxies = {
        scheme: str(system_proxies[scheme]).strip()
        for scheme in ("http", "https")
        if str(system_proxies.get(scheme) or "").strip()
    }
    if proxies:
        client_config["proxies"] = proxies
    ex = ccxt.bybit(client_config)
    ex.set_sandbox_mode(cfg.is_testnet)
    _assert_endpoints(cfg, ex)
    return ex


def _assert_endpoints(cfg: BybitConfig, exchange: Any) -> None:
    """Fail closed unless every ccxt API endpoint resolves to the profile's host.

    The host is the authoritative paper/live discriminator: testnet keys cannot
    reach the live host. Reading the client's own resolved URLs (rather than
    the config's declared host) catches a sandbox switch that did not apply.
    """
    urls = _obj_get(_obj_get(exchange, "urls", {}), "api", {})
    if not isinstance(urls, Mapping) or not urls:
        raise BybitConfigError("Bybit client exposes no API endpoints to verify")
    expected = cfg.host
    for name, url in urls.items():
        resolved = str(url).replace("{hostname}", str(_obj_get(exchange, "hostname", cfg.hostname)))
        host = (urlparse(resolved).hostname or "").lower()
        if host != expected:
            raise BybitConfigError(
                f"Configured profile is {cfg.environment}, but the '{name}' endpoint resolves to "
                f"'{host}' instead of '{expected}'."
            )


def _missing_fields(cfg: BybitConfig) -> list[str]:
    missing = []
    if not cfg.api_key:
        missing.append("api_key")
    if not cfg.api_secret:
        missing.append("api_secret")
    return missing


def _public_config(cfg: BybitConfig) -> dict[str, Any]:
    """Config snapshot with secrets redacted."""
    data = asdict(cfg)
    if data.get("api_secret"):
        data["api_secret"] = "***redacted***"
    if data.get("api_key"):
        data["api_key"] = data["api_key"][:4] + "***"
    data["host"] = cfg.host
    return data
