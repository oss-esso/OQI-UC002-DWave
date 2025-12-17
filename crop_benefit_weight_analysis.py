#!/usr/bin/env python3
"""
Crop Benefit Weight Analysis
=============================
Analyzes how crop benefits change across all weight combinations that sum to 1.

The benefit formula is:
    B_c = w1 * nutritional_value + w2 * nutrient_density - w3 * environmental_impact 
          + w4 * affordability + w5 * sustainability

Where: w1 + w2 + w3 + w4 + w5 = 1

Author: GitHub Copilot
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from itertools import product
from typing import Dict, List, Tuple
import os
from tqdm import tqdm

# Import unified plot configuration
from plot_config import (
    setup_publication_style, QUALITATIVE_COLORS, METHOD_COLORS,
    FOOD_GROUP_COLORS, save_figure, add_value_labels
)

# Apply publication style
setup_publication_style()


def load_food_data(excel_path: str = "Inputs/Combined_Food_Data.xlsx") -> pd.DataFrame:
    """Load food data from Excel file."""
    df = pd.read_excel(excel_path)
    return df


def calculate_benefit(food_data: Dict, weights: Dict, subtract_env: bool = True) -> float:
    """
    Calculate composite benefit score for a food.
    
    Args:
        food_data: Dictionary with food attributes
        weights: Dictionary with weight values (w1-w5)
        subtract_env: If True, subtract environmental impact (lower is better)
    
    Returns:
        Benefit score
    """
    if subtract_env:
        benefit = (
            weights['nutritional_value'] * food_data['nutritional_value'] +
            weights['nutrient_density'] * food_data['nutrient_density'] -
            weights['environmental_impact'] * food_data['environmental_impact'] +
            weights['affordability'] * food_data['affordability'] +
            weights['sustainability'] * food_data['sustainability']
        )
    else:
        # Alternative: treat all as positive contributions
        benefit = (
            weights['nutritional_value'] * food_data['nutritional_value'] +
            weights['nutrient_density'] * food_data['nutrient_density'] +
            weights['environmental_impact'] * (1 - food_data['environmental_impact']) +
            weights['affordability'] * food_data['affordability'] +
            weights['sustainability'] * food_data['sustainability']
        )
    return benefit


def generate_weight_combinations(step: float = 0.1, n_weights: int = 5) -> List[Tuple]:
    """
    Generate all weight combinations that sum to 1.
    
    Args:
        step: Step size for weight values (e.g., 0.1 for 0, 0.1, 0.2, ... 1.0)
        n_weights: Number of weights (default 5)
    
    Returns:
        List of tuples, each containing a valid weight combination
    """
    # Generate possible values for each weight
    values = np.arange(0, 1 + step/2, step)
    values = np.round(values, 2)  # Avoid floating point issues
    
    valid_combinations = []
    
    # For 5 weights, we need to find all combinations that sum to 1
    for combo in tqdm(product(values, repeat=n_weights), 
                      desc="Generating weight combinations",
                      total=len(values)**n_weights):
        if abs(sum(combo) - 1.0) < 1e-9:
            valid_combinations.append(combo)
    
    return valid_combinations


def analyze_all_combinations(df: pd.DataFrame, step: float = 0.1) -> pd.DataFrame:
    """
    Analyze crop benefits for all weight combinations.
    
    Returns:
        DataFrame with columns: weight combo, food names, benefits, ranking
    """
    weight_names = ['nutritional_value', 'nutrient_density', 'environmental_impact', 
                    'affordability', 'sustainability']
    
    # Generate all valid weight combinations
    combinations = generate_weight_combinations(step=step)
    print(f"\nFound {len(combinations)} valid weight combinations")
    
    # Build food data dictionary
    foods = {}
    for _, row in df.iterrows():
        foods[row['Food_Name']] = {
            'nutritional_value': row['nutritional_value'],
            'nutrient_density': row['nutrient_density'],
            'environmental_impact': row['environmental_impact'],
            'affordability': row['affordability'],
            'sustainability': row['sustainability'],
            'food_group': row['food_group']
        }
    
    # Results storage
    results = []
    
    for combo in tqdm(combinations, desc="Calculating benefits"):
        weights = dict(zip(weight_names, combo))
        
        # Calculate benefits for all foods
        benefits = {}
        for food_name, food_data in foods.items():
            benefits[food_name] = calculate_benefit(food_data, weights)
        
        # Sort by benefit (highest first)
        sorted_foods = sorted(benefits.items(), key=lambda x: x[1], reverse=True)
        
        # Store results
        results.append({
            'weights': combo,
            'w_nutr_val': combo[0],
            'w_nutr_den': combo[1],
            'w_env_imp': combo[2],
            'w_afford': combo[3],
            'w_sustain': combo[4],
            'top_crop': sorted_foods[0][0],
            'top_benefit': sorted_foods[0][1],
            'rankings': {food: rank+1 for rank, (food, _) in enumerate(sorted_foods)},
            'benefits': benefits
        })
    
    return pd.DataFrame(results)


def plot_top_crop_distribution(results_df: pd.DataFrame, save_path: str = None):
    """Plot bar chart showing how often each crop is #1."""
    top_counts = results_df['top_crop'].value_counts()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(top_counts)))
    bars = ax.barh(range(len(top_counts)), top_counts.values, color=colors)
    
    ax.set_yticks(range(len(top_counts)))
    ax.set_yticklabels(top_counts.index, fontsize=11)
    ax.set_xlabel('Number of Weight Combinations Where Crop is #1', fontsize=12)
    ax.set_title('🏆 Which Crop Wins Most Often?\n(Across All Weight Combinations)', 
                 fontsize=14, fontweight='bold')
    
    # Add count labels on bars
    for bar, count in zip(bars, top_counts.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{count}', va='center', fontsize=10)
    
    ax.invert_yaxis()  # Highest count at top
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    plt.close(fig)


def plot_benefit_heatmap(results_df: pd.DataFrame, foods_list: List[str], 
                         n_samples: int = 50, save_path: str = None):
    """
    Heatmap showing benefits across sampled weight combinations.
    """
    # Sample combinations evenly
    sample_indices = np.linspace(0, len(results_df)-1, n_samples, dtype=int)
    sampled = results_df.iloc[sample_indices]
    
    # Build matrix: rows = foods, cols = weight combos
    benefit_matrix = np.zeros((len(foods_list), n_samples))
    
    for j, (_, row) in enumerate(sampled.iterrows()):
        for i, food in enumerate(foods_list):
            benefit_matrix[i, j] = row['benefits'][food]
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    im = ax.imshow(benefit_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    
    ax.set_yticks(range(len(foods_list)))
    ax.set_yticklabels(foods_list)
    ax.set_xlabel('Weight Combination Index (sampled)', fontweight='bold')
    ax.set_ylabel('Crop', fontweight='bold')
    ax.set_title('Crop Benefits Across Weight Combinations', fontweight='bold', pad=20)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Benefit Score', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    plt.close(fig)


def plot_ranking_variability(results_df: pd.DataFrame, foods_list: List[str], 
                             save_path: str = None):
    """Box plot showing ranking variability for each crop."""
    ranking_data = {food: [] for food in foods_list}
    
    for _, row in results_df.iterrows():
        for food in foods_list:
            ranking_data[food].append(row['rankings'][food])
    
    # Calculate mean ranking for sorting
    mean_rankings = {food: np.mean(ranks) for food, ranks in ranking_data.items()}
    sorted_foods = sorted(mean_rankings.keys(), key=lambda x: mean_rankings[x])
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create box plot data
    box_data = [ranking_data[food] for food in sorted_foods]
    
    bp = ax.boxplot(box_data, vert=False, patch_artist=True, 
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    medianprops=dict(linewidth=2, color='darkred'))
    
    # Color boxes using qualitative palette
    n_foods = len(sorted_foods)
    colors = [QUALITATIVE_COLORS[i % len(QUALITATIVE_COLORS)] for i in range(n_foods)]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_yticklabels(sorted_foods)
    ax.set_xlabel('Ranking (1 = Best, 27 = Worst)', fontweight='bold')
    ax.set_title('Crop Ranking Variability Across Weight Combinations', fontweight='bold', pad=20)
    
    # Add vertical line at median
    ax.axvline(x=14, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Middle rank (14)')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    plt.close(fig)


def plot_weight_sensitivity(results_df: pd.DataFrame, focus_crops: List[str], 
                            weight_name: str = 'w_nutr_val', save_path: str = None):
    """
    Line plot showing how benefit changes with one weight (averaging over others).
    """
    weight_labels = {
        'w_nutr_val': 'Nutritional Value Weight',
        'w_nutr_den': 'Nutrient Density Weight', 
        'w_env_imp': 'Environmental Impact Weight',
        'w_afford': 'Affordability Weight',
        'w_sustain': 'Sustainability Weight'
    }
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use consistent qualitative colors
    colors = [QUALITATIVE_COLORS[i % len(QUALITATIVE_COLORS)] for i in range(len(focus_crops))]
    
    for crop, color in zip(focus_crops, colors):
        # Group by the weight value and calculate mean benefit
        weight_vals = []
        mean_benefits = []
        std_benefits = []
        
        for w_val in results_df[weight_name].unique():
            subset = results_df[results_df[weight_name] == w_val]
            benefits = [row['benefits'][crop] for _, row in subset.iterrows()]
            weight_vals.append(w_val)
            mean_benefits.append(np.mean(benefits))
            std_benefits.append(np.std(benefits))
        
        # Sort by weight value
        sorted_idx = np.argsort(weight_vals)
        weight_vals = np.array(weight_vals)[sorted_idx]
        mean_benefits = np.array(mean_benefits)[sorted_idx]
        std_benefits = np.array(std_benefits)[sorted_idx]
        
        ax.plot(weight_vals, mean_benefits, 'o-', label=crop, color=color, linewidth=2, markersize=6)
        ax.fill_between(weight_vals, mean_benefits - std_benefits, mean_benefits + std_benefits, 
                        alpha=0.2, color=color)
    
    ax.set_xlabel(weight_labels[weight_name], fontweight='bold')
    ax.set_ylabel('Average Benefit Score', fontweight='bold')
    ax.set_title(f'Sensitivity to {weight_labels[weight_name]}', fontweight='bold', pad=20)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    plt.close(fig)


def plot_spinach_dominance_analysis(results_df: pd.DataFrame, df: pd.DataFrame, 
                                    save_path: str = None):
    """
    Special analysis: When and why does Spinach dominate?
    """
    spinach_wins = results_df[results_df['top_crop'] == 'Spinach']
    spinach_loses = results_df[results_df['top_crop'] != 'Spinach']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    weight_cols = ['w_nutr_val', 'w_nutr_den', 'w_env_imp', 'w_afford', 'w_sustain']
    weight_labels = ['Nutritional\nValue', 'Nutrient\nDensity', 'Environmental\nImpact', 
                     'Affordability', 'Sustainability']
    
    # Top row: Weight distributions when Spinach wins vs loses
    for i, (col, label) in enumerate(zip(weight_cols, weight_labels)):
        ax = axes[0, i] if i < 3 else axes[1, i-3]
        
        if len(spinach_wins) > 0:
            ax.hist(spinach_wins[col], alpha=0.6, label='Spinach #1', 
                    color=QUALITATIVE_COLORS[2], bins=10, density=True, edgecolor='black', linewidth=0.5)
        if len(spinach_loses) > 0:
            ax.hist(spinach_loses[col], alpha=0.6, label='Other #1', 
                    color=QUALITATIVE_COLORS[0], bins=10, density=True, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel(label, fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        ax.legend()
    
    # Bottom right: Spinach attributes bar chart
    ax_radar = axes[1, 2]
    spinach_data = df[df['Food_Name'] == 'Spinach'].iloc[0]
    attrs = ['nutritional_value', 'nutrient_density', 'environmental_impact', 
             'affordability', 'sustainability']
    values = [spinach_data[attr] for attr in attrs]
    
    # Use consistent color palette
    colors_bar = QUALITATIVE_COLORS[:len(attrs)]
    bars = ax_radar.bar(range(len(attrs)), values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax_radar.set_xticks(range(len(attrs)))
    ax_radar.set_xticklabels(['Nutr. Value', 'Nutr. Density', 'Env. Impact', 
                              'Afford.', 'Sustain.'])
    ax_radar.set_ylabel('Score (0-1)', fontweight='bold')
    ax_radar.set_title('Spinach Attributes', fontweight='bold', pad=10)
    ax_radar.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax_radar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                      f'{val:.2f}', ha='center', fontsize=8)
    
    fig.suptitle('Why Spinach Dominates: Weight Distribution Analysis', fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    plt.close(fig)


def plot_parallel_coordinates(results_df: pd.DataFrame, top_n: int = 10, 
                              save_path: str = None):
    """
    Parallel coordinates plot showing weight combinations where different crops win.
    Manual implementation without pandas to avoid seaborn dependency.
    """
    # Get the most common winning crops
    top_winners = results_df['top_crop'].value_counts().head(top_n).index.tolist()
    
    # Filter to these crops
    filtered = results_df[results_df['top_crop'].isin(top_winners)].copy()
    
    # Sample for readability
    if len(filtered) > 500:
        filtered = filtered.sample(n=500, random_state=42)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Normalize weights to 0-1 for plotting
    weight_cols = ['w_nutr_val', 'w_nutr_den', 'w_env_imp', 'w_afford', 'w_sustain']
    x_positions = list(range(len(weight_cols)))
    
    # Color by crop
    crop_colors = {crop: QUALITATIVE_COLORS[i % len(QUALITATIVE_COLORS)] 
                   for i, crop in enumerate(top_winners)}
    
    # Plot lines for each weight combination
    for _, row in filtered.iterrows():
        y_vals = [row[col] for col in weight_cols]
        color = crop_colors[row['top_crop']]
        ax.plot(x_positions, y_vals, color=color, alpha=0.3, linewidth=1)
    
    # Customize axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['Nutr. Value', 'Nutr. Density', 'Env. Impact', 
                        'Affordability', 'Sustainability'])
    ax.set_ylabel('Weight Value', fontweight='bold')
    ax.set_title('Weight Combinations by Winning Crop', fontweight='bold', pad=20)
    ax.set_ylim(-0.05, 1.05)
    
    # Create legend
    handles = [plt.Line2D([0], [0], color=crop_colors[crop], linewidth=2, label=crop) 
               for crop in top_winners]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    plt.close(fig)


def create_summary_table(results_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Create a summary table with crop statistics."""
    foods_list = df['Food_Name'].tolist()
    
    summary_data = []
    
    for food in foods_list:
        # Get ranking statistics
        rankings = [row['rankings'][food] for _, row in results_df.iterrows()]
        benefits = [row['benefits'][food] for _, row in results_df.iterrows()]
        
        # Count wins
        wins = (results_df['top_crop'] == food).sum()
        
        # Food group
        food_group = df[df['Food_Name'] == food]['food_group'].values[0]
        
        summary_data.append({
            'Crop': food,
            'Food Group': food_group,
            'Times #1': wins,
            'Win Rate (%)': round(100 * wins / len(results_df), 2),
            'Best Rank': min(rankings),
            'Worst Rank': max(rankings),
            'Mean Rank': round(np.mean(rankings), 2),
            'Rank Std': round(np.std(rankings), 2),
            'Mean Benefit': round(np.mean(benefits), 4),
            'Benefit Std': round(np.std(benefits), 4)
        })
    
    summary = pd.DataFrame(summary_data)
    summary = summary.sort_values('Mean Rank')
    
    return summary


def print_extreme_cases(results_df: pd.DataFrame):
    """Print the weight combinations for extreme cases."""
    print("\n" + "="*70)
    print("🔍 EXTREME CASES ANALYSIS")
    print("="*70)
    
    # Best case for Spinach
    spinach_best = results_df.loc[results_df.apply(
        lambda x: x['benefits']['Spinach'], axis=1).idxmax()]
    print(f"\n✅ Highest Spinach Benefit ({spinach_best['benefits']['Spinach']:.4f}):")
    print(f"   Weights: NV={spinach_best['w_nutr_val']:.1f}, ND={spinach_best['w_nutr_den']:.1f}, "
          f"EI={spinach_best['w_env_imp']:.1f}, AF={spinach_best['w_afford']:.1f}, "
          f"SU={spinach_best['w_sustain']:.1f}")
    
    # Worst case for Spinach (but still top 3)
    spinach_ranks = results_df.apply(lambda x: x['rankings']['Spinach'], axis=1)
    spinach_worst_idx = spinach_ranks.idxmax()
    spinach_worst = results_df.loc[spinach_worst_idx]
    print(f"\n⚠️ Worst Spinach Ranking (#{spinach_worst['rankings']['Spinach']}):")
    print(f"   Weights: NV={spinach_worst['w_nutr_val']:.1f}, ND={spinach_worst['w_nutr_den']:.1f}, "
          f"EI={spinach_worst['w_env_imp']:.1f}, AF={spinach_worst['w_afford']:.1f}, "
          f"SU={spinach_worst['w_sustain']:.1f}")
    print(f"   Winner: {spinach_worst['top_crop']} with benefit {spinach_worst['top_benefit']:.4f}")
    
    # Cases where Spinach is NOT #1
    non_spinach = results_df[results_df['top_crop'] != 'Spinach']
    if len(non_spinach) > 0:
        print(f"\n📊 Spinach is NOT #1 in {len(non_spinach)} of {len(results_df)} combinations "
              f"({100*len(non_spinach)/len(results_df):.1f}%)")
        
        # Who beats Spinach?
        beaters = non_spinach['top_crop'].value_counts()
        print("   Crops that beat Spinach:")
        for crop, count in beaters.items():
            print(f"      {crop}: {count} times ({100*count/len(non_spinach):.1f}%)")


def main():
    """Main function to run the analysis."""
    print("="*70)
    print("🌾 CROP BENEFIT WEIGHT ANALYSIS")
    print("   Analyzing all 27 crops across weight combinations")
    print("="*70)
    
    # Create output directory
    output_dir = "crop_weight_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\n📂 Loading food data...")
    df = load_food_data()
    foods_list = df['Food_Name'].tolist()
    print(f"   Loaded {len(foods_list)} crops")
    
    # Analyze all combinations (step=0.1 gives manageable number)
    print("\n🔢 Generating and analyzing weight combinations...")
    results_df = analyze_all_combinations(df, step=0.1)
    
    # Create summary table
    print("\n📋 Creating summary table...")
    summary = create_summary_table(results_df, df)
    print("\n" + summary.to_string())
    summary.to_csv(f"{output_dir}/crop_ranking_summary.csv", index=False)
    print(f"\n   Saved: {output_dir}/crop_ranking_summary.csv")
    
    # Print extreme cases
    print_extreme_cases(results_df)
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    
    # 1. Top crop distribution
    plot_top_crop_distribution(results_df, f"{output_dir}/01_top_crop_distribution.png")
    
    # 2. Benefit heatmap
    plot_benefit_heatmap(results_df, foods_list, n_samples=100, 
                         save_path=f"{output_dir}/02_benefit_heatmap.png")
    
    # 3. Ranking variability
    plot_ranking_variability(results_df, foods_list, 
                             save_path=f"{output_dir}/03_ranking_variability.png")
    
    # 4. Weight sensitivity for top crops
    top_crops = summary.head(8)['Crop'].tolist()
    for weight in ['w_nutr_val', 'w_nutr_den', 'w_env_imp', 'w_afford', 'w_sustain']:
        plot_weight_sensitivity(results_df, top_crops, weight_name=weight,
                                save_path=f"{output_dir}/04_sensitivity_{weight}.png")
    
    # 5. Spinach dominance analysis
    plot_spinach_dominance_analysis(results_df, df, 
                                    save_path=f"{output_dir}/05_spinach_analysis.png")
    
    # 6. Parallel coordinates
    plot_parallel_coordinates(results_df, top_n=8, 
                              save_path=f"{output_dir}/06_parallel_coordinates.png")
    
    print("\n" + "="*70)
    print("✅ Analysis complete!")
    print(f"   Results saved to: {output_dir}/")
    print("="*70)
    
    return results_df, summary


if __name__ == "__main__":
    results_df, summary = main()
