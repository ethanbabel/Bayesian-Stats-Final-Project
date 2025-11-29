from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm, spearmanr

from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
OFF_FEATURE_COLUMNS: Sequence[str] = ["0-3", "3-10", "10-16", "16-3P"]
DEF_FEATURE_COLUMNS: Sequence[str] = ["def_0-3", "def_3-10", "def_10-16", "def_16-3P"]
FEATURE_COLUMNS: Sequence[str] = [*OFF_FEATURE_COLUMNS, *DEF_FEATURE_COLUMNS]
DIAGNOSTIC_FEATURE_COLUMNS: Sequence[str] = [
    *OFF_FEATURE_COLUMNS,
    "3P",
    *DEF_FEATURE_COLUMNS,
    "def_3P",
]
SEASONS: Sequence[str] = ["2004-05", "2024-25"]
N_GAMES = 82

# Weak normal priors: broad on the intercept, moderately broad on slopes
PRIOR_SD_INTERCEPT = 5.0
PRIOR_SD_SLOPE = 2.5

@dataclass
class SamplerResult:
    draws: np.ndarray
    acceptance_rate: float
    log_posterior_trace: np.ndarray
    param_names: Sequence[str]
    season: str

def load_cleaned(season: str) -> pd.DataFrame:
    path = DATA_DIR / f"cleaned_data_{season}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cleaned data not found for season {season} at {path}")
    return pd.read_csv(path)

def build_design(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    required = list(FEATURE_COLUMNS) + ["3P", "def_3P", "wins"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["wins"].to_numpy(dtype=float)
    return X, y

def log_likelihood(params: np.ndarray, X: np.ndarray, wins: np.ndarray, n_games: int) -> float:
    alpha = params[0]
    betas = params[1:]
    logits = alpha + X @ betas
    # log(p) = log(sigmoid(logits)) = log(1 / (1 + exp(-logits))) = -log(1 + exp(-logits))
    log_p = -np.logaddexp(0.0, -logits)
    # log(1 - p) = log(1 - sigmoid(logits)) = log(sigmoid(-logits)) = log(1 / (1 + exp(logits))) = -log(1 + exp(logits))
    log_1_minus_p = -np.logaddexp(0.0, logits)
    ll = wins * log_p + (n_games - wins) * log_1_minus_p
    return float(np.sum(ll))

def log_prior(params: np.ndarray) -> float:
    alpha = params[0]
    betas = params[1:]
    lp_alpha = norm.logpdf(alpha, loc=0.0, scale=PRIOR_SD_INTERCEPT)
    lp_betas = norm.logpdf(betas, loc=0.0, scale=PRIOR_SD_SLOPE).sum()
    return float(lp_alpha + lp_betas)

def log_posterior(params: np.ndarray, X: np.ndarray, wins: np.ndarray, n_games: int) -> float:
    return log_likelihood(params, X, wins, n_games) + log_prior(params)

def rw_metropolis(
    log_post: Callable[[np.ndarray], float],
    initial: np.ndarray,
    step_scales: np.ndarray,
    num_steps: int,
    burn_in: int,
    seed: int = 42,
) -> tuple[np.ndarray, float, np.ndarray]:
    print("Starting Metropolis sampling...")

    rng = np.random.default_rng(seed)
    params = initial.copy()
    current_lp = log_post(params)

    draws: list[np.ndarray] = []
    trace_lp: list[float] = []
    accepts = 0

    for step in tqdm(range(num_steps)):
        proposal = params + rng.normal(scale=step_scales, size=params.shape)
        proposal_lp = log_post(proposal)
        log_accept_ratio = proposal_lp - current_lp
        if np.log(rng.uniform()) < log_accept_ratio:
            params = proposal
            current_lp = proposal_lp
            accepts += 1

        trace_lp.append(current_lp)
        if step >= burn_in:
            draws.append(params.copy())

    acceptance_rate = accepts / num_steps
    return np.vstack(draws), acceptance_rate, np.array(trace_lp)

def summarize_draws(draws: np.ndarray, param_names: Sequence[str]) -> pd.DataFrame:
    quantiles = np.quantile(draws, [0.025, 0.5, 0.975], axis=0)
    summary = pd.DataFrame(
        {
            "mean": draws.mean(axis=0),
            "sd": draws.std(axis=0, ddof=1),
            "q2.5": quantiles[0],
            "median": quantiles[1],
            "q97.5": quantiles[2],
        },
        index=param_names,
    )
    return summary

def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    x_centered = x - x.mean()
    denom = np.dot(x_centered, x_centered)
    acf = [1.0]
    for lag in range(1, max_lag + 1):
        num = np.dot(x_centered[:-lag], x_centered[lag:])
        acf.append(num / denom if denom != 0 else 0.0)
    return np.array(acf)

def effective_sample_size(draws: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    n, _ = draws.shape
    max_lag = max_lag or min(1000, n // 2)
    ess = []
    for j in range(draws.shape[1]):
        acf = autocorrelation(draws[:, j], max_lag)
        
        rho_sum = 0.0
        for k in range(1, len(acf), 2):
            pair_sum = acf[k] + (acf[k + 1] if k + 1 < len(acf) else 0.0)
            if pair_sum < 0:
                break # Stop adding when a lag pair turns negative
            rho_sum += pair_sum
        ess.append(n / (1 + 2 * rho_sum))
    return np.array(ess)

def plot_trace_and_acf(
    draws: np.ndarray,
    param_names: Sequence[str],
    season: str,
    output_dir: Path,
    max_lag: int = 1000,
) -> None:
    for idx, name in enumerate(param_names):
        series = draws[:, idx]
        lags = np.arange(0, max_lag + 1)
        acf_vals = autocorrelation(series, max_lag)

        fig, axes = plt.subplots(2, 1, figsize=(7, 5), constrained_layout=True)
        axes[0].plot(series, lw=0.8)
        axes[0].set_title(f"{name} trace ({season})")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Value")

        axes[1].stem(lags, acf_vals, basefmt=" ")
        axes[1].set_title(f"{name} autocorrelation")
        axes[1].set_xlabel("Lag")
        axes[1].set_ylabel("ACF")

        fig.savefig(output_dir / f"{season}_{name}_trace_acf.png", dpi=150)
        plt.close(fig)

def residual_correlations(residuals: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in DIAGNOSTIC_FEATURE_COLUMNS:
        feature = df[col].to_numpy(dtype=float)
        pearson = float(np.corrcoef(feature, residuals)[0, 1])
        spearman, _ = spearmanr(feature, residuals)
        rows.append({"feature": col, "pearson_r": pearson, "spearman_r": float(spearman)})
    return pd.DataFrame(rows).set_index("feature")

def plot_residuals(
    df: pd.DataFrame,
    residuals: np.ndarray,
    std_residuals: np.ndarray,
    season: str,
    output_dir: Path,
) -> None:
    for col in DIAGNOSTIC_FEATURE_COLUMNS:
        x = df[col].to_numpy(dtype=float)

        fig, axes = plt.subplots(2, 1, figsize=(7, 6), constrained_layout=True)

        axes[0].scatter(x, residuals, alpha=0.7, s=22)
        axes[0].axhline(0.0, color="gray", ls="--", lw=1)
        axes[0].set_title(f"Residual vs {col} ({season})")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Residual (obs - pred)")

        axes[1].scatter(x, std_residuals, alpha=0.7, s=22, color="tab:orange")
        axes[1].axhline(0.0, color="gray", ls="--", lw=1)
        axes[1].set_title("Standardized residuals")
        axes[1].set_xlabel(col)
        axes[1].set_ylabel("Std residual")

        fig.savefig(output_dir / f"{season}_residuals_{col}.png", dpi=150)
        plt.close(fig)

def fit_season(
    season: str,
    num_steps: int = 105_000,
    burn_in: int = 25_000,
    step_scale_intercept: float = 0.2,
    step_scale_slope: float = 0.2,
) -> SamplerResult:
    df = load_cleaned(season)
    X, wins = build_design(df)

    initial = np.zeros(X.shape[1] + 1)
    step_scales = np.full_like(initial, step_scale_slope, dtype=float)
    step_scales[0] = step_scale_intercept

    log_post = lambda params: log_posterior(params, X, wins, N_GAMES)
    draws, acc_rate, lp_trace = rw_metropolis(
        log_post=log_post,
        initial=initial,
        step_scales=step_scales,
        num_steps=num_steps,
        burn_in=burn_in,
    )
    return SamplerResult(
        draws=draws,
        acceptance_rate=acc_rate,
        log_posterior_trace=lp_trace,
        param_names=["alpha"] + list(FEATURE_COLUMNS),
        season=season,
    )

def main() -> None:
    for season in SEASONS:
        print(f"Fitting season {season}...")

        result = fit_season(season)
        summary = summarize_draws(result.draws, result.param_names)
        print(f"\nSeason {season}")
        print(f"Acceptance rate: {result.acceptance_rate:.3f}")
        print(summary.round(3))
        print("Effective sample size:")
        ess = effective_sample_size(result.draws)
        ess_df = pd.Series(ess, index=result.param_names, name="ess")
        print(ess_df.round(1))
        plot_trace_and_acf(
            result.draws,
            result.param_names,
            season=season,
            output_dir=OUTPUT_DIR,
        )
        print(f"Diagnostics saved to {OUTPUT_DIR}")
        print()

        # Posterior mean win probability per team (using posterior mean parameters)
        df = load_cleaned(season)
        X, wins = build_design(df)
        mean_params = result.draws.mean(axis=0)
        logits = mean_params[0] + X @ mean_params[1:]
        win_probs = expit(logits)
        df_out = df[["team", "wins"]].copy()
        df_out["posterior_mean_win_prob"] = win_probs
        df_out["posterior_mean_wins"] = win_probs * N_GAMES
        # Brier score - mean squared error between predicted win probability and observed win rate
        observed_win_rate = wins / N_GAMES
        brier = float(np.mean((win_probs - observed_win_rate) ** 2))
        baseline_p = float(observed_win_rate.mean())
        brier_baseline = float(np.mean((baseline_p - observed_win_rate) ** 2))
        skill = 1.0 - (brier / brier_baseline if brier_baseline > 0 else np.nan)
        print(f"Brier score: {brier:.4f} | Baseline: {brier_baseline:.4f} | Skill: {skill:.3f}")
        # Residual diagnostics: observed win rate minus predicted win probability
        residuals = observed_win_rate - win_probs
        var = win_probs * (1 - win_probs) / N_GAMES
        std_residuals = residuals / np.sqrt(var + 1e-12)
        corr_df = residual_correlations(residuals, df)
        print("Residual correlations with shot shares (Pearson/Spearman):")
        print(corr_df.round(3))
        plot_residuals(df, residuals, std_residuals, season=season, output_dir=OUTPUT_DIR)
        # print(df_out.sort_values("posterior_mean_win_prob", ascending=False))

if __name__ == "__main__":
    main()
