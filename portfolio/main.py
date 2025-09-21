from typing import List, Dict, Optional
import time
from datetime import datetime
from pathlib import Path

import numba as nb
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import click


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
            valid_tickers = set(portfolio_composition_dict.keys())
            self.cached_df = self.cached_df.drop(
                columns=[
                    ticker
                    for ticker in self.cached_df.columns
                    if ticker not in valid_tickers
                ]
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
) -> np.ndarray:
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


@nb.njit
def portfolio_stats(W, ret, cov_mat, risk_free_rate=0.0):
    """calculate portfolio info for each weight"""
    n, m = W.shape
    expected_returns = np.empty(n, dtype=np.float64)
    volatilities = np.empty(n, dtype=np.float64)
    sharpe_ratios = np.empty(n, dtype=np.float64)

    for i in range(n):
        # portfolio return
        port_ret = 0.0
        for j in range(m):
            port_ret += W[i, j] * ret[j]
        expected_returns[i] = port_ret

        port_var = 0.0
        for j in range(m):
            for k in range(m):
                port_var += W[i, j] * cov_mat[j, k] * W[i, k]

        port_vol = np.sqrt(port_var)
        volatilities[i] = port_vol

        if port_vol > 0.0:
            sharpe_ratios[i] = (port_ret - risk_free_rate) / port_vol
        else:
            sharpe_ratios[i] = -np.inf

    return expected_returns, volatilities, sharpe_ratios


def plot_portfolios(
    weights,
    portfolio_returns,
    portfolio_volatilities,
    sharpe_ratios,
    output_path: Path = Path(f"./data/{datetime.now()}_portfolio_output.png"),
):
    """Plot the resulting portfolios with best sharp ratio"""
    results = pd.DataFrame(
        {
            "expected_return": portfolio_returns * 100,
            "volatility": portfolio_volatilities * 100,
            "sharpe_ratio": sharpe_ratios,
        }
    )

    best_idx = results["sharpe_ratio"].idxmax()
    best = results.loc[best_idx]

    best_portfolio_ticker_weights = {
        key: val for key, val in zip(PORTFOLIO_COMPOSITION.keys(), weights[best_idx])
    }
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=results,
        x="volatility",
        y="expected_return",
        hue="sharpe_ratio",
        palette="viridis",
        alpha=0.7,
        s=20,
    )
    plt.scatter(
        best["volatility"],
        best["expected_return"],
        color="red",
        s=200,
        label=f'Best Sharpe: {best["sharpe_ratio"]:.3f}',
    )
    weights_text = "Optimal Weights:\n" + "\n".join(
        [
            f"• {ticker}: {weight:.2%}"
            for ticker, weight in best_portfolio_ticker_weights.items()
        ]
    )
    plt.title("Monte Carlo Portfolio Optimization", fontsize=16)
    plt.text(
        0.02,
        0.98,
        weights_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5", fc="white", alpha=0.6
        ),  # Add a semi-transparent box
    )
    plt.xlabel("Volatility in percentage %"), plt.ylabel(
        "Expected Return in percentage %"
    ), plt.legend()

    plt.savefig(output_path)

    return best, best_idx


# S&P 500 (^GSPC) : free weights
# Gold Dec 25 (GC=F) : free weights
# Emerging markets iShares MSCI EM UCITS ETF USD (Acc) (IEMA.L) : free weights = > Removed: high correlation to US and India
# Amundi MSCI China UCITS ETF Acc (LCCN.L) : free weights
# iShares U.S. Oil & Gas Exploration & Production ETF (IEO)
# iShares Silver Trust (SLV) => pretty high correlation to gold
# Global X DAX Germany ETF (DAX)
# iShares Bitcoin Trust ETF (IBIT)
# Vanguard FTSE Europe ETF (VGK) : Stopped : Too hight
# Vanguard Short-Term Bond Index Fund ETF Shares (BSV)
# iShares MSCI World ETF (URTH) => Removed: 0.97 corrolation to SNP
# Global X Copper Miners ETF (COPX)
# iShares MSCI United Kingdom ETF (EWU) => Removed: 0.76 corrolation to SNP, High average correlation
# iShares MSCI India ETF (INDA)
# iShares 20+ Year Treasury Bond ETF (TLT)


RAW_DATA_FILE_PATH = Path("./data/portfolio_raw_data.xlsx")
PROCESSED_DATA_FILE_PATH = Path("./data/portfolio_processing.xlsx")

PORTFOLIO_COMPOSITION = {
    "^GSPC": 0.0,
    "GC=F": 0.0,
    "IEO": 0.0,
    "DAX": 0.0,
    "BTC-USD": 0.0,
    "COPX": 0.0,
    "INDA": 0.0,
    "TLT": 0.0,
}


@click.command()
@click.option("--risk_free_rate", default=0.0425, type=float)
@click.option("--n_rand_portfolios", default=200000, type=int)
@click.option("--max_volatility", default=np.inf, type=float)
def main(risk_free_rate, n_rand_portfolios, max_volatility):
    """Main entrypoint for script"""
    writer = pd.ExcelWriter(PROCESSED_DATA_FILE_PATH, engine="xlsxwriter")
    ticker_manager = TickerManager(RAW_DATA_FILE_PATH, PORTFOLIO_COMPOSITION)

    weekly_returns_df = ticker_manager.get_returns_df()
    weekly_returns_df.to_excel(writer, sheet_name="full_data")

    weekly_returns_df = weekly_returns_df.dropna()
    weekly_returns_df.to_excel(writer, sheet_name="data_no_na")

    weekly_returns_percent_df = weekly_returns_df.pct_change().dropna()
    weekly_returns_percent_df.to_excel(writer, sheet_name="percent_returns")

    covariance_matrix_array = np.asarray(weekly_returns_percent_df.cov() * 52)
    weekly_returns_percent_df.corr().to_excel(writer, sheet_name="correlation_matrix")

    writer.close()

    weights = generate_random_weights(PORTFOLIO_COMPOSITION, n_rand_portfolios)

    yearly_average_returns_array = np.asarray(
        ((1 + weekly_returns_percent_df.mean()) ** 52 - 1)
    )

    expected_returns, volatilities, sharpe_ratios = portfolio_stats(
        weights, yearly_average_returns_array, covariance_matrix_array, risk_free_rate
    )

    assert (
        volatilities.min() < max_volatility
    ), f"Min calculated volatility ({volatilities.min()*100:.2F}%) is over the defined allowed Max vol.({max_volatility*100:.2F}%)"

    mask = volatilities < max_volatility
    expected_returns, volatilities, sharpe_ratios, weights = (
        expected_returns[mask],
        volatilities[mask],
        sharpe_ratios[mask],
        weights[mask],
    )

    plot_portfolios(weights, expected_returns, volatilities, sharpe_ratios)


if __name__ == "__main__":
    main()
