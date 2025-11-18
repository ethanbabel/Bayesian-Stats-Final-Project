import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath, skiprows=1)
    
    teams = df.iloc[:, 1]  # Team column
    fga_2p = df.iloc[:, 7]   # 2P column
    fga_0_3 = df.iloc[:, 8]   # 0-3 column  
    fga_3_10 = df.iloc[:, 9]  # 3-10 column
    fga_10_16 = df.iloc[:, 10] # 10-16 column
    fga_16_3p = df.iloc[:, 11] # 16-3P column
    fga_3p = df.iloc[:, 12]    # 3P column
    
    clean_df = pd.DataFrame({
        'Team': teams,
        '2P': pd.to_numeric(fga_2p, errors='coerce') * 100,
        '0-3': pd.to_numeric(fga_0_3, errors='coerce') * 100,
        '3-10': pd.to_numeric(fga_3_10, errors='coerce') * 100,
        '10-16': pd.to_numeric(fga_10_16, errors='coerce') * 100,
        '16-3P': pd.to_numeric(fga_16_3p, errors='coerce') * 100,
        '3P': pd.to_numeric(fga_3p, errors='coerce') * 100
    })
    
    clean_df = clean_df.dropna().reset_index(drop=True)
    
    return clean_df

def create_scatterplot(df, season, output_dir):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {
        '0-3': '#1f77b4',
        '3-10': '#ff7f0e',
        '10-16': '#2ca02c',
        '16-3P': '#d62728',
        '3P': '#9467bd'
    }
    
    
    distance_ranges = ['0-3', '3-10', '10-16', '16-3P', '3P']
    
    for i, distance in enumerate(distance_ranges):
        x_values = [i] * len(df)
        y_values = df[distance].values
        
        # Add some jitter to x-values to separate overlapping points
        x_jittered = x_values + np.random.normal(0, 0.1, len(x_values))
        
        ax.scatter(x_jittered, y_values, c=colors[distance], alpha=0.7, s=60, label=f'{distance} ft')
    
    ax.set_xlabel('Shot Distance Range', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Field Goal Attempts (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Distribution of FGA by Distance - {season} Season', fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xticks(range(len(distance_ranges)))
    ax.set_xticklabels(distance_ranges)
    
    ax.grid(True, alpha=0.3)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    output_path = output_dir / f'fga_by_distance_{season.replace("-", "_")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Scatterplot saved to: {output_path}")

def create_comparison_plot(df_2004, df_2024, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    colors = {
        '0-3': '#1f77b4',
        '3-10': '#ff7f0e',
        '10-16': '#2ca02c',
        '16-3P': '#d62728',
        '3P': '#9467bd'
    }
    
    distance_ranges = ['0-3', '3-10', '10-16', '16-3P', '3P']
    
    # Plot 2004-05 season
    for i, distance in enumerate(distance_ranges):
        x_values = [i] * len(df_2004)
        y_values = df_2004[distance].values
        x_jittered = x_values + np.random.normal(0, 0.1, len(x_values))
        
        ax1.scatter(x_jittered, y_values, c=colors[distance], alpha=0.7, s=60)
    
    ax1.set_xlabel('Shot Distance Range', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Percentage of Field Goal Attempts (%)', fontsize=12, fontweight='bold')
    ax1.set_title('2004-05 Season', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(distance_ranges)))
    ax1.set_xticklabels(distance_ranges)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(df_2004[distance_ranges].max().max(), df_2024[distance_ranges].max().max()) * 1.1)
    
    # Plot 2024-25 season
    for i, distance in enumerate(distance_ranges):
        x_values = [i] * len(df_2024)
        y_values = df_2024[distance].values
        x_jittered = x_values + np.random.normal(0, 0.1, len(x_values))
        
        ax2.scatter(x_jittered, y_values, c=colors[distance], alpha=0.7, s=60, label=f'{distance} ft')
    
    ax2.set_xlabel('Shot Distance Range', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percentage of Field Goal Attempts (%)', fontsize=12, fontweight='bold')
    ax2.set_title('2024-25 Season', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(distance_ranges)))
    ax2.set_xticklabels(distance_ranges)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(df_2004[distance_ranges].max().max(), df_2024[distance_ranges].max().max()) * 1.1)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle('NBA Shot Distribution Comparison: 2004-05 vs 2024-25', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'fga_comparison_2004_vs_2024.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Comparison plot saved to: {output_path}")

if __name__ == "__main__":
    data_dir = Path('data')
    output_dir = Path('prep/plots')
    output_dir.mkdir(exist_ok=True)
    
    df_2004 = load_and_clean_data(data_dir / 'stats_2004-05.csv')
    df_2024 = load_and_clean_data(data_dir / 'stats_2024-25.csv')
    
    create_scatterplot(df_2004, "2004-05", output_dir)
    create_scatterplot(df_2024, "2024-25", output_dir)
    
    create_comparison_plot(df_2004, df_2024, output_dir)
    