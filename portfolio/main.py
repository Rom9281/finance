from typing import List, Dict, Optional
import time
from datetime import datetime
from pathlib import Path

import numba
import yfinance as yf
import pandas as pd
import numpy as np


class TickerManager:
    def __init__(
        self,
        cached_returns_file_path: Path,
        portfolio_composition_dict: Dict[str, float],
    ) -> None:
        self.portfolio_composition_dict = portfolio_composition_dict
        self.cached_returns_file_path = cached_returns_file_path
        self.cached_df = pd.DataFrame()
        try:
            self.cached_df = pd.read_excel(self.cached_returns_file_path).set_index(
                "Date"
            )
        except Exception as e:
            print(f"Error: {e}; Downloading all values")
            cached_returns_file_path.parent.mkdir(exist_ok=True, parents=True)

    def _get_missing_tickers(self) -> List[str]:
        """returns the list of ticker to dl, otherwise the whole list of ticker"""
        if not self.cached_df.empty:
            return [
                ticker
                for ticker in self.portfolio_composition_dict.keys()
                if ticker not in set(self.cached_df.columns)
            ]
        return list(self.portfolio_composition_dict.keys())

    def get_tickers_data(self, ticker_list: List[str]) -> List[pd.DataFrame]:
        "download the tickers from yfinance"
        tickers_close_data_df_list = []
        for ticker in ticker_list:
            try:
                time.sleep(1)
                ticker_history = yf.Tickers(ticker).history(
                    period="max", interval="1wk"
                )
                tickers_close_data_df_list.append(ticker_history["Close"])
            except Exception as e:
                print(f"Error with {ticker} : {e}")
        return pd.concat(tickers_close_data_df_list, axis=1)

    def get_returns_df(self, save_to_excel: bool = True, drop_na=False) -> pd.DataFrame:
        """get the returns from the object"""
        missing_tickers_list = self._get_missing_tickers()
        if missing_tickers_list:
            additional_returns_df = self.get_tickers_data(missing_tickers_list)
            merged_cache_dl_df = pd.concat(
                [self.cached_df, additional_returns_df], axis=1
            )
            if save_to_excel:
                merged_cache_dl_df.to_excel(self.cached_returns_file_path)
            if drop_na:
                merged_cache_dl_df = merged_cache_dl_df.dropna()
            return merged_cache_dl_df
        return self.cached_df


def get_constrained_indices(portfolio_constraints: List[float]) -> List[int]:
    """Get indices for weights that are non zero"""
    return [i for i, weight in enumerate(portfolio_constraints) if weight > 0.0]


def generate_random_weights(
    portfolio_composition: Dict[str, float], n_rand_portfolios: int = 50000
):
    """Generates random weights taking into account constraints"""
    portfolio_constraints = portfolio_composition.values()

    n_constrained_items = len(get_constrained_indices(portfolio_constraints))
    n_free_weights = len(portfolio_composition) - n_constrained_items

    if n_free_weights == 0:
        return np.tile(portfolio_constraints, (n_rand_portfolios, 1))

    constrained_weight_sum = sum(portfolio_constraints)
    free_weights_sum = 1 - constrained_weight_sum

    if free_weights_sum < 0:
        raise ValueError(f"Fixed weights sum to {constrained_weight_sum} > 1")

    free_weights_vect = (
        np.random.dirichlet(np.ones(n_free_weights), n_rand_portfolios)
        * free_weights_sum
    )

    i = 0
    weight_vector_list = []

    for weight in portfolio_constraints:
        if 1.0 > weight > 0.0:
            weight_vector_list.append(np.full(n_rand_portfolios, weight))
        elif weight == 0.0:
            weight_vector_list.append(free_weights_vect[:, i])
            i += 1
        else:
            raise ValueError("Weight must be between [0;1[")
    return np.column_stack(weight_vector_list)


# S&P 500 (^GSPC) : free weights
# Gold Dec 25 (GC=F) : free weights
# Emerging markets iShares MSCI EM UCITS ETF USD (Acc) (IEMA.L) : free weights
# iShares Physical Gold ETC (IGLN.L)

FILE_PATH = Path("./portfolio.xlsx")

PORTFOLIO_COMPOSITION = {
    "^GSPC":0.0,
    "GC=F":0.0,
    "IEMA.L":0.0,
    
}

RISK_FREE_RATE = 4.25

def main():
    ticker_manager = TickerManager(FILE_PATH, PORTFOLIO_COMPOSITION)
    returns_df = ticker_manager.get_returns_df()
    print(returns_df.head())
    print(generate_random_weights(PORTFOLIO_COMPOSITION))


if __name__ == "__main__":
    main()
