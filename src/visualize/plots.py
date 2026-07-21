"""
Visualization module for RUL prediction results.
Generates: degradation curves, confidence intervals, t-SNE plots.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')


def plot_rul_prediction(y_true, y_pred, y_std=None, title="RUL Prediction", 
                         save_path=None):
    """Plot true vs predicted RUL with optional confidence intervals."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n = len(y_true)
    x = np.arange(n)
    
    # Scatter
    ax.scatter(x, y_true, c='blue', s=30, alpha=0.7, label='True RUL', zorder=3)
    ax.scatter(x, y_pred, c='red', s=30, alpha=0.7, marker='x', label='Predicted RUL', zorder=3)
    
    # Confidence intervals
    if y_std is not None:
        y_std = np.array(y_std)
        ax.fill_between(x, y_pred - 1.96*y_std, y_pred + 1.96*y_std,
                        alpha=0.2, color='red', label='95% CI')
    
    # Perfect prediction line
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Perfect')
    
    ax.set_xlabel('Test Sample')
    ax.set_ylabel('Remaining Useful Life (cycles)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    path = save_path or os.path.join(OUTPUT_DIR, 'rul_prediction.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")
    return path


def plot_degradation_trajectory(times, health_index, rul_pred=None, 
                                 failure_threshold=None, title="Degradation Trajectory",
                                 save_path=None):
    """Plot degradation trajectory over time with RUL prediction."""
    fig, ax1 = plt.subplots(figsize=(12, 5))
    
    # Health index
    color = 'tab:blue'
    ax1.plot(times, health_index, color=color, linewidth=2, label='Health Index')
    ax1.set_xlabel('Time (cycles)')
    ax1.set_ylabel('Health Index', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Failure threshold
    if failure_threshold is not None:
        ax1.axhline(y=failure_threshold, color='red', linestyle='--', 
                    alpha=0.7, label='Failure Threshold')
    
    # RUL prediction
    if rul_pred is not None:
        ax2 = ax1.twinx()
        ax2.plot(times, rul_pred, color='orange', linewidth=2, linestyle='--',
                 label='Predicted RUL')
        ax2.set_ylabel('Predicted RUL (cycles)', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels() if rul_pred is not None else ([],[])
    ax1.legend(lines1+lines2, labels1+labels2, loc='upper right')
    
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    
    path = save_path or os.path.join(OUTPUT_DIR, 'degradation.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return path


def plot_confidence_bands(times, rul_mean, total_std, title="RUL with Uncertainty",
                           save_path=None):
    """Plot RUL prediction with confidence bands over time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    times = np.array(times)
    rul_mean = np.array(rul_mean)
    total_std = np.array(total_std)
    
    # Confidence bands
    ax.fill_between(times, rul_mean - 2*total_std, rul_mean + 2*total_std,
                    alpha=0.1, color='blue', label='95% CI')
    ax.fill_between(times, rul_mean - total_std, rul_mean + total_std,
                    alpha=0.2, color='blue', label='68% CI')
    
    # Mean prediction
    ax.plot(times, rul_mean, 'b-', linewidth=2, label='Predicted RUL')
    
    # Diagonal reference
    ax.plot([times[0], times[-1]], [times[-1]-times[0], 0], 'k--', 
            alpha=0.3, label='Ideal')
    
    ax.set_xlabel('Time (cycles)')
    ax.set_ylabel('Remaining Useful Life (cycles)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    path = save_path or os.path.join(OUTPUT_DIR, 'confidence.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return path


def plot_tsne(features, labels=None, domains=None, title="t-SNE Feature Visualization",
              save_path=None):
    """t-SNE visualization of encoded features."""
    # Sample if too many points
    n = len(features)
    if n > 1000:
        idx = np.random.choice(n, 1000, replace=False)
        features = features[idx]
        if labels is not None: labels = labels[idx]
        if domains is not None: domains = domains[idx]
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n-1))
    embedded = tsne.fit_transform(features)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if domains is not None:
        for d in np.unique(domains):
            mask = domains == d
            ax.scatter(embedded[mask, 0], embedded[mask, 1], s=10, alpha=0.6, label=f'Domain {d}')
    elif labels is not None:
        scatter = ax.scatter(embedded[:, 0], embedded[:, 1], c=labels, cmap='viridis',
                            s=10, alpha=0.6)
        plt.colorbar(scatter, label='RUL')
    else:
        ax.scatter(embedded[:, 0], embedded[:, 1], s=10, alpha=0.6)
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(title)
    if domains is not None: ax.legend()
    
    path = save_path or os.path.join(OUTPUT_DIR, 'tsne.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return path


def plot_comparison_bar(results_dict, metric='RMSE', title="Method Comparison",
                         save_path=None):
    """Bar chart comparing methods across datasets."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    datasets = list(results_dict.keys())
    methods = list(results_dict[datasets[0]].keys())
    
    x = np.arange(len(datasets))
    width = 0.8 / len(methods)
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']
    
    for i, method in enumerate(methods):
        values = [results_dict[d][method] for d in datasets]
        ax.bar(x + i*width, values, width, label=method, color=colors[i % len(colors)])
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_xticks(x + width * (len(methods)-1) / 2)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    path = save_path or os.path.join(OUTPUT_DIR, 'comparison.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return path


# Demo
if __name__ == '__main__':
    # Generate sample data for testing
    np.random.seed(42)
    n = 50
    
    # RUL prediction
    y_true = np.linspace(130, 0, n) + np.random.randn(n) * 5
    y_pred = y_true + np.random.randn(n) * 10
    y_std = np.abs(np.random.randn(n) * 8)
    plot_rul_prediction(y_true, y_pred, y_std, "Sample RUL Prediction")
    
    # Degradation
    t = np.arange(100)
    hi = 1 - 0.01*t - 0.0002*t**2 + np.random.randn(100)*0.02
    rul = 100 - t + np.random.randn(100)*5
    plot_degradation_trajectory(t, hi, rul, failure_threshold=0.2)
    
    # Confidence
    plot_confidence_bands(np.arange(50), np.linspace(200, 0, 50), np.linspace(5, 15, 50))
    
    # t-SNE
    X = np.random.randn(200, 64)
    labs = np.random.randint(0, 130, 200)
    plot_tsne(X, labs, title="Sample t-SNE")
    
    # Comparison
    results = {
        'FD001': {'Pretrained': 18.5, 'Random': 42.1, 'LSTM': 16.2, 'Wiener': 25.3},
        'FD002': {'Pretrained': 22.1, 'Random': 45.3, 'LSTM': 19.8, 'Wiener': 28.7},
        'FD003': {'Pretrained': 20.3, 'Random': 43.5, 'LSTM': 17.9, 'Wiener': 26.1},
    }
    plot_comparison_bar(results)
    
    print("All demo plots generated in outputs/")
