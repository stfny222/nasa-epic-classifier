"""
Generate Comparison Visualizations for All Experiments
=======================================================

Creates publication-quality plots comparing all 4 model architectures:
- Baseline CNN
- Multi-Scale CNN  
- ResNet18 Transfer Learning
- Ensemble (ResNet + Multi-Scale)
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# Define paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

EXPERIMENTS = {
    'Baseline CNN': BASE_DIR / 'baseline_cnn' / 'outputs',
    'Multi-Scale CNN': BASE_DIR / 'multiscale_cnn' / 'outputs',
    'ResNet18 Transfer': BASE_DIR / 'resnet_transfer' / 'outputs',
}

LABEL_NAMES = ['North America', 'South America', 'Europe', 'Africa', 'Asia', 'Oceania']


def load_experiment_data():
    """Load metrics and training history from all experiments."""
    data = {}
    
    for name, path in EXPERIMENTS.items():
        # Load test metrics
        metrics_file = path / 'test_metrics.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Load training history
            history_file = path / 'training_history.csv'
            if history_file.exists():
                history = pd.read_csv(history_file)
            else:
                history = None
            
            data[name] = {
                'metrics': metrics,
                'history': history
            }
    
    # Load ensemble results
    ensemble_file = BASE_DIR / 'ensemble' / 'outputs' / 'ensemble_results.json'
    if ensemble_file.exists():
        with open(ensemble_file, 'r') as f:
            ensemble_data = json.load(f)
        # Use the best strategy (Config Weights)
        data['Ensemble (Config)'] = {
            'metrics': ensemble_data['config_weights'],
            'history': None
        }
    
    return data


def plot_1_overall_metrics_comparison(data):
    """Bar chart comparing overall F1, Precision, Recall."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = list(data.keys())
    metrics_to_plot = [
        ('f1_macro', 'F1 Score (Macro)', axes[0]),
        ('precision_macro', 'Precision (Macro)', axes[1]),
        ('recall_macro', 'Recall (Macro)', axes[2])
    ]
    
    for metric_key, title, ax in metrics_to_plot:
        values = [data[model]['metrics'][metric_key] for model in models]
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(models)]
        
        bars = ax.bar(range(len(models)), values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.set_ylabel('Score')
        ax.set_title(title, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '1_overall_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 1_overall_metrics_comparison.png")
    plt.close()


def plot_2_per_label_performance(data):
    """Heatmap showing F1 score per label for each model."""
    models = list(data.keys())
    
    # Extract per-label F1 scores
    f1_matrix = []
    for model in models:
        per_label = data[model]['metrics'].get('per_label', {})
        f1_scores = [per_label.get(label, {}).get('f1', 0.0) for label in LABEL_NAMES]
        f1_matrix.append(f1_scores)
    
    f1_matrix = np.array(f1_matrix)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(f1_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(range(len(LABEL_NAMES)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels(LABEL_NAMES, rotation=45, ha='right')
    ax.set_yticklabels(models)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('F1 Score', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(LABEL_NAMES)):
            text = ax.text(j, i, f'{f1_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Per-Continent F1 Score Comparison', fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '2_per_label_f1_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 2_per_label_f1_heatmap.png")
    plt.close()


def plot_3_training_curves(data):
    """Training and validation curves for models with history."""
    models_with_history = {k: v for k, v in data.items() if v['history'] is not None}
    
    if not models_with_history:
        print("⚠ No training history found, skipping training curves")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for idx, (model_name, model_data) in enumerate(models_with_history.items()):
        history = model_data['history']
        color = colors[idx % len(colors)]
        
        # Plot loss only (accuracy columns may vary between experiments)
        ax.plot(history['train_loss'], label=f'{model_name} (Train)', 
                color=color, linewidth=2, alpha=0.8)
        ax.plot(history['val_loss'], label=f'{model_name} (Val)', 
                color=color, linewidth=2, linestyle='--', alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss Over Time', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '3_training_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 3_training_curves.png")
    plt.close()


def plot_4_radar_chart(data):
    """Radar chart showing per-label F1 for best model."""
    # Use ResNet18 as the best model
    best_model = 'ResNet18 Transfer'
    if best_model not in data:
        best_model = list(data.keys())[0]
    
    per_label = data[best_model]['metrics'].get('per_label', {})
    f1_scores = [per_label.get(label, {}).get('f1', 0.0) for label in LABEL_NAMES]
    
    # Radar chart setup
    angles = np.linspace(0, 2 * np.pi, len(LABEL_NAMES), endpoint=False).tolist()
    f1_scores += f1_scores[:1]  # Close the loop
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.plot(angles, f1_scores, 'o-', linewidth=2, color='#2ecc71', label=best_model)
    ax.fill(angles, f1_scores, alpha=0.25, color='#2ecc71')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('F1 Score', labelpad=30)
    ax.set_title(f'Per-Continent Performance: {best_model}', 
                 fontweight='bold', pad=20, fontsize=16)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '4_best_model_radar.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 4_best_model_radar.png")
    plt.close()


def plot_5_improvement_over_baseline(data):
    """Bar chart showing percentage improvement over baseline."""
    baseline_name = 'Baseline CNN'
    if baseline_name not in data:
        print("⚠ Baseline CNN not found, skipping improvement chart")
        return
    
    baseline_f1 = data[baseline_name]['metrics']['f1_macro']
    
    models = [m for m in data.keys() if m != baseline_name]
    improvements = []
    
    for model in models:
        model_f1 = data[model]['metrics']['f1_macro']
        improvement = ((model_f1 - baseline_f1) / baseline_f1) * 100
        improvements.append(improvement)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c', '#2ecc71', '#f39c12'][:len(models)]
    bars = ax.barh(models, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Improvement over Baseline (%)', fontweight='bold')
    ax.set_title('F1 Score Improvement Relative to Baseline CNN', fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
               f'{width:+.1f}%', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '5_improvement_over_baseline.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 5_improvement_over_baseline.png")
    plt.close()


def generate_summary_table(data):
    """Generate markdown table with all metrics."""
    rows = []
    
    for model_name, model_data in data.items():
        metrics = model_data['metrics']
        rows.append({
            'Model': model_name,
            'F1 (Macro)': f"{metrics['f1_macro']:.4f}",
            'Precision': f"{metrics['precision_macro']:.4f}",
            'Recall': f"{metrics['recall_macro']:.4f}",
            'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
            'Hamming Loss': f"{metrics.get('hamming_loss', 0):.4f}"
        })
    
    df = pd.DataFrame(rows)
    
    # Save as CSV
    df.to_csv(OUTPUT_DIR / 'summary_table.csv', index=False)
    print(f"✓ Saved: summary_table.csv")
    
    # Save as markdown (manual formatting without tabulate dependency)
    with open(OUTPUT_DIR / 'summary_table.md', 'w') as f:
        f.write("# Model Performance Summary\n\n")
        
        # Header
        headers = df.columns.tolist()
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join([" --- " for _ in headers]) + "|\n")
        
        # Rows
        for _, row in df.iterrows():
            f.write("| " + " | ".join(str(v) for v in row.values) + " |\n")
    
    print(f"✓ Saved: summary_table.md")
    
    return df


def main():
    """Generate all visualizations."""
    print("=" * 70)
    print("GENERATING EXPERIMENT COMPARISON VISUALIZATIONS")
    print("=" * 70)
    
    # Load data
    print("\n[1/6] Loading experiment data...")
    data = load_experiment_data()
    print(f"  Loaded {len(data)} experiments")
    
    # Generate plots
    print("\n[2/6] Generating overall metrics comparison...")
    plot_1_overall_metrics_comparison(data)
    
    print("\n[3/6] Generating per-label F1 heatmap...")
    plot_2_per_label_performance(data)
    
    print("\n[4/6] Generating training curves...")
    plot_3_training_curves(data)
    
    print("\n[5/6] Generating radar chart for best model...")
    plot_4_radar_chart(data)
    
    print("\n[6/6] Generating improvement comparison...")
    plot_5_improvement_over_baseline(data)
    
    # Generate summary table
    print("\n[Bonus] Generating summary table...")
    df = generate_summary_table(data)
    
    print("\n" + "=" * 70)
    print("✓ All visualizations generated successfully!")
    print("=" * 70)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.glob('*')):
        print(f"  - {file.name}")
    
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
