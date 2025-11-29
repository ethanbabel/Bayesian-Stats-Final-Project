from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
PERCENT_COLUMNS = ['2P', '0-3', '3-10', '10-16', '16-3P', '3P']

def _clean_team(series: pd.Series) -> pd.Series:
    return series.str.replace('*', '', regex=False).str.strip()

def load_standings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=1)
    df = df[['Team', 'Overall']].copy()
    df['team'] = _clean_team(df['Team'])
    wins_losses = df['Overall'].str.split('-', n=1, expand=True)
    df['wins'] = pd.to_numeric(wins_losses[0], errors='coerce')
    df['losses'] = pd.to_numeric(wins_losses[1], errors='coerce')
    return df[['team', 'wins', 'losses']].dropna(subset=['team'])

def load_distance_stats(path: Path, prefix: str = '') -> pd.DataFrame:
    df = pd.read_csv(path, header=1)
    df = df[['Team'] + PERCENT_COLUMNS].copy()
    df['team'] = _clean_team(df['Team'])
    for col in PERCENT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if prefix:
        renamed = {col: f"{prefix}{col}" for col in PERCENT_COLUMNS}
        df = df.rename(columns=renamed)
        cols = ['team'] + [renamed[col] for col in PERCENT_COLUMNS]
    else:
        cols = ['team'] + PERCENT_COLUMNS
    return df[cols]

def build_output() -> pd.DataFrame:
    seasons = [
        (
            '2004-05',
            DATA_DIR / 'standings_2004-05.csv',
            DATA_DIR / 'offense_stats_2004-05.csv',
            DATA_DIR / 'defense_stats_2004-05.csv',
        ),
        (
            '2024-25',
            DATA_DIR / 'standings_2024-25.csv',
            DATA_DIR / 'offense_stats_2024-25.csv',
            DATA_DIR / 'defense_stats_2024-25.csv',
        ),
    ]
    for season, standings_path, offense_path, defense_path in seasons:
        standings = load_standings(standings_path)
        offense = load_distance_stats(offense_path)
        defense = load_distance_stats(defense_path, prefix='def_')
        merged = pd.merge(standings, offense, on='team', how='inner')
        merged = pd.merge(merged, defense, on='team', how='inner')
        merged.to_csv(DATA_DIR / f'cleaned_data_{season}.csv', index=False)
        print(f'Cleaned data for {season} saved.')

if __name__ == '__main__':
    build_output()
