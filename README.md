# NBA Shot Distribution & Winning: A Bayesian Model
Bayesian modeling of NBA team win totals using shot profile data (both offensive and defensive) across two seasons, with residual diagnostics and feature contribution plots.

## Usage
- Install dependencies: `pip install -r requirements.txt`
- Build cleaned season inputs from raw CSVs: `python scripts/clean_data.py` (writes `data/cleaned_data_*.csv`)
- Fit models and generate diagnostics/plots: `python scripts/bayesian_model.py` (writes to `output/`)

## Repo Structure
- `scripts/clean_data.py` – assembles season CSVs into cleaned modeling tables
- `scripts/bayesian_model.py` – MCMC fit, diagnostics, plots, cross-validation
- `data/` – raw season inputs and generated `cleaned_data_*.csv`
- `output/` – posterior summaries, residual plots, contribution charts
- `latex/` – full write-up (`latex/report.pdf`), including latex source (`latex/report.tex`)
- `prep/` – exploratitive notebooks/plots for project prep

## Full Report and Slide Deck
See the detailed methodology and results: 
- `Bayesian_FinalProject_Report.pdf`
- `Bayesian_FinalProject_Slides.pdf`
