Bayesian NBA Win Model
======================

This project fits a simple Bayesian model to predict NBA team wins from shot-profile data (offensive and defensive distance splits) and then decomposes predicted wins into feature-level contributions. It supports two seasons out of the box: 2004-05 and 2024-25.

Project structure
-----------------
- `data/`: raw CSV inputs (standings and shot distance splits) and cleaned outputs.
- `scripts/clean_data.py`: builds season-level cleaned datasets.
- `scripts/bayesian_model.py`: fits the Bayesian model, produces diagnostics, and saves predictions/plots.
- `output/`: generated artifacts
  - `plots/`: feature contribution bar charts (average team)
  - `trace_acf/`: trace and autocorrelation plots per parameter
  - `residuals_analysis/`: residual vs feature plots
  - season-level CSVs (posterior win probabilities, feature contributions)

Data cleaning
-------------
Run `python scripts/clean_data.py` to merge standings with offensive and defensive shot-distance percentages for each season, producing `data/cleaned_data_{season}.csv` with columns:
- `team`, `wins`, `losses`
- Offense shot share percentages: `0-3`, `3-10`, `10-16`, `16-3P`, `3P`
- Defense shot share percentages: `def_0-3`, `def_3-10`, `def_10-16`, `def_16-3P`, `def_3P`

Bayesian model
--------------
- Observation model: For team *i*, wins 
$$
w_i ~ Binomial(n_games=82, p_i)
$$
$$
p_i = \sigma (α + Σ_j β_j x_{ij})
$$
- Features: 8 distance-split shares (`0-3`, `3-10`, `10-16`, `16-3P` and defensive counterparts). The omitted 3P buckets are incorporated in contributions via compositional contrasts (implied β for 3P = negative sum of the other offense betas; same for defense).
- Standardization: Each feature is z-scored using season-level means and std devs before modeling.
- Priors: `α ~ Normal(0, 0.5)`; each `β_j ~ Normal(0, 0.1)`.
- Inference: Random-walk Metropolis-Hastings with Gaussian proposals.
  - Default settings: `num_steps=105_000`, `burn_in=25_000`, `step_scale_intercept=0.05`, `step_scale_slope=0.05`, RNG seed 2025.
  - Outputs posterior draws, acceptance rate, and log-posterior trace.
- Diagnostics: Effective sample size per parameter, trace and ACF plots, residual correlations with shot shares, and leave-one-out cross-validation skill (Brier).

Win contributions
-----------------
Using posterior mean parameters, predicted wins per team are decomposed into contributions:
1) Standardize features with the same means/stds used in fitting.
2) Compute linear terms for offense/defense splits and implied 3P contrasts.
3) Allocate each feature's share of predicted wins proportionally to its linear-term weight (including the intercept).

Plot: The contribution plot now shows a single bar per feature for the **average team** (mean contribution across teams), centered at zero so positive values extend upward and negative values downward. File: `output/plots/{season}_feature_contributions.png`.

Running the model
-----------------
1) Install dependencies (Python 3.11+ recommended):
   ```
   pip install numpy pandas matplotlib scipy tqdm
   ```
2) Clean data (if not already present):
   ```
   python scripts/clean_data.py
   ```
3) Fit model and generate outputs:
   ```
   python scripts/bayesian_model.py
   ```
   Key outputs land in `output/` as CSVs and plots; console prints posterior summaries and CV skill.

Key outputs (per season)
------------------------
- `output/{season}_posterior_win_probs.csv`: posterior mean win probabilities and wins per team.
- `output/{season}_feature_contributions.csv`: per-team win contributions by feature (and intercept).
- `output/plots/{season}_feature_contributions.png`: average-team feature contribution bars.
- `output/plots/{season}_posterior_forest.png`: posterior band effects (mean ± 95% CI).
- `output/plots/{season}_pred_vs_actual.png`: scatter of predicted vs actual win rates.
- `output/trace_acf/*`: parameter trace + autocorrelation plots.
- `output/residuals_analysis/*`: residual diagnostics vs features.

Reproducing / extending
-----------------------
- To add a season: place matching standings/offense/defense CSVs in `data/`, mirror the naming pattern, and add an entry in `scripts/clean_data.py::build_output()`.
- To tweak priors or sampler settings: adjust constants near the top of `scripts/bayesian_model.py`.
- To change the model form (e.g., different features or hierarchical pooling), modify `FEATURE_COLUMNS` and `log_likelihood`/`log_prior` accordingly.
