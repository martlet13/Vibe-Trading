"""HK runs must ask the loader for a benchmark in project (suffix) form.

Loaders take ``700.HK``-style symbols; a Futu-style ``HK.03100`` passes
through the yfinance/Longbridge symbol mappers unchanged, Yahoo answers
"No data found", and every HK backtest silently loses its benchmark.
"""

import pandas as pd
import pytest

from backtest.benchmark import resolve_benchmark
from backtest.loaders.yfinance_loader import _to_yfinance_symbol


@pytest.mark.parametrize("symbol", ["700.HK", "00700.HK", "9988.HK"])
def test_hk_strategy_benchmarks_against_hscei_etf_in_suffix_form(symbol):
    seen = []

    class Loader:
        name = "yfinance"

        def fetch(self, codes, *args, **kwargs):
            seen.extend(codes)
            return {
                codes[0]: pd.DataFrame(
                    {"close": [10.0, 11.0]},
                    index=pd.date_range("2025-01-01", periods=2),
                )
            }

    result = resolve_benchmark(
        [symbol], "yfinance", "2025-01-01", "2025-01-02", loader=Loader()
    )

    assert result is not None
    assert seen == [result.ticker]
    assert _to_yfinance_symbol(result.ticker) == "2828.HK"
