"""
Visualization module for AF-FCL research metrics.

Generates publication-quality plots using matplotlib and seaborn:

  1. Accuracy vs Tasks        — line plot of avg accuracy after each task
  2. Forgetting vs Tasks      — bar chart of per-task forgetting
  3. Communication Cost       — cumulative line + per-round bar
  4. Bias Metric Comparison   — grouped bar chart of fairness metrics
  5. Cosine Similarity Dist.  — histogram + KDE
  6. Accuracy Heatmap         — A[t][i] heatmap
  7. Cosine vs Forgetting     — scatter + correlation
  8. Cosine vs Accuracy       — scatter + correlation

All plots use a consistent professional style with tight layouts.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/SSH
import matplotlib.pyplot as plt
import seaborn as sns
import glog as logger


# ======================================================================
#  GLOBAL STYLE
# ======================================================================

def _setup_style():
    """Configure consistent publication-quality plot style."""
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.2)
    plt.rcParams.update({
        'figure.figsize': (8, 5),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 2,
        'lines.markersize': 6,
    })


# Color palette
COLORS = sns.color_palette('deep', 10)


# ======================================================================
#  1. ACCURACY VS TASKS
# ======================================================================

def plot_accuracy_vs_tasks(tracker, save_path: str):
    """
    Line plot of average accuracy after learning each task.

    Shows both global accuracy and per-task accuracy trends.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    num_tasks = tracker.num_tasks
    acc_matrix = tracker.accuracy_matrix

    # Global accuracy per task
    glob_accs = []
    for t in range(num_tasks):
        row = acc_matrix[t, :t + 1]
        valid = row[~np.isnan(row)]
        glob_accs.append(np.mean(valid) * 100 if len(valid) > 0 else 0.0)

    tasks = list(range(num_tasks))
    ax.plot(tasks, glob_accs, 'o-', color=COLORS[0], label='Avg Accuracy',
            linewidth=2.5, markersize=8, zorder=5)

    # Per-task accuracy evolution (final evaluation)
    for i in range(num_tasks):
        col = acc_matrix[:, i]
        # Only plot from when task i first appears
        task_accs = []
        task_xs = []
        for t in range(i, num_tasks):
            if not np.isnan(col[t]):
                task_xs.append(t)
                task_accs.append(col[t] * 100)
        if len(task_accs) > 0:
            ax.plot(task_xs, task_accs, '--', alpha=0.5, linewidth=1,
                    label=f'Task {i}' if i < 6 else None)

    ax.set_xlabel('Task Learned')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy vs Tasks')
    ax.set_xticks(tasks)
    ax.legend(loc='best', framealpha=0.9)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# ======================================================================
#  2. FORGETTING VS TASKS
# ======================================================================

def plot_forgetting_vs_tasks(tracker, save_path: str):
    """Bar chart of per-task forgetting and average forgetting line."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    forgetting = tracker.get_forgetting_per_task()
    tasks = list(range(len(forgetting)))
    avg_f = np.mean(forgetting)

    bars = ax.bar(tasks, forgetting, color=COLORS[3], alpha=0.8, edgecolor='black',
                  linewidth=0.5, label='Per-task Forgetting')

    # Average line
    ax.axhline(y=avg_f, color=COLORS[1], linestyle='--', linewidth=2,
               label=f'Avg Forgetting = {avg_f:.4f}')

    # Value labels on bars
    for bar, val in zip(bars, forgetting):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Task')
    ax.set_ylabel('Forgetting')
    ax.set_title('Catastrophic Forgetting per Task')
    ax.set_xticks(tasks)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# ======================================================================
#  3. COMMUNICATION COST
# ======================================================================

def plot_communication_cost(tracker, save_path: str):
    """Dual-axis plot: per-round cost (bar) + cumulative cost (line)."""
    _setup_style()

    if len(tracker.comm_rounds) == 0:
        logger.warning("No communication data to plot")
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    rounds = [r['round_idx'] for r in tracker.comm_rounds]
    per_round = [r['round_total_mb'] for r in tracker.comm_rounds]
    cumulative = np.cumsum(per_round)

    # Per-round bars
    ax1.bar(rounds, per_round, color=COLORS[0], alpha=0.6, label='Per Round (MB)')

    # Cumulative line
    ax2.plot(rounds, cumulative, 'o-', color=COLORS[3], linewidth=2,
             markersize=4, label='Cumulative (MB)')

    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Per-Round Cost (MB)', color=COLORS[0])
    ax2.set_ylabel('Cumulative Cost (MB)', color=COLORS[3])
    ax1.set_title('Communication Cost Over Rounds')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# ======================================================================
#  4. BIAS METRIC COMPARISON
# ======================================================================

def plot_bias_comparison(tracker, save_path: str):
    """Grouped bar chart of fairness metrics."""
    _setup_style()

    if not tracker.fairness_results:
        logger.warning("No fairness data to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    metric_names = ['demographic_parity', 'equality_of_opportunity',
                    'equalized_odds_tpr_diff', 'equalized_odds_fpr_diff',
                    'predictive_parity']
    display_names = ['Demographic\nParity', 'Equality of\nOpportunity',
                     'Equalized Odds\n(TPR Diff)', 'Equalized Odds\n(FPR Diff)',
                     'Predictive\nParity']

    values = []
    valid_names = []
    valid_display = []
    for m, d in zip(metric_names, display_names):
        if m in tracker.fairness_results:
            values.append(tracker.fairness_results[m])
            valid_names.append(m)
            valid_display.append(d)

    if len(values) == 0:
        plt.close(fig)
        return

    x = np.arange(len(values))
    bars = ax.bar(x, values, color=[COLORS[i % len(COLORS)] for i in range(len(values))],
                  alpha=0.8, edgecolor='black', linewidth=0.5)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Fairness Metric')
    ax.set_ylabel('Disparity (lower = fairer)')
    ax.set_title('Bias / Fairness Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_display)
    ax.grid(True, alpha=0.3, axis='y')

    # Reference line at 0
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# ======================================================================
#  5. COSINE SIMILARITY DISTRIBUTION
# ======================================================================

def plot_cosine_distribution(tracker, save_path: str):
    """Histogram + KDE of cosine similarity scores across all tasks."""
    _setup_style()

    if not tracker.cosine_stats_per_task:
        logger.warning("No cosine data to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Collect per-batch means from all tasks
    all_means = []
    for task_idx in sorted(tracker.cosine_stats_per_task.keys()):
        stats = tracker.cosine_stats_per_task[task_idx]
        batch_means = stats.get('per_batch_means', [])
        if batch_means:
            all_means.extend(batch_means)

    if len(all_means) == 0:
        plt.close(fig)
        return

    all_means = np.array(all_means)

    # Histogram + KDE
    ax.hist(all_means, bins=50, density=True, alpha=0.6, color=COLORS[2],
            edgecolor='black', linewidth=0.3, label='Histogram')

    # KDE overlay
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(all_means)
        x_range = np.linspace(all_means.min() - 0.1, all_means.max() + 0.1, 200)
        ax.plot(x_range, kde(x_range), color=COLORS[3], linewidth=2.5,
                label='KDE')
    except ImportError:
        pass  # scipy not available, skip KDE

    ax.axvline(x=np.mean(all_means), color=COLORS[1], linestyle='--',
               linewidth=2, label=f'Mean = {np.mean(all_means):.4f}')

    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.set_title('Cosine Similarity Distribution (Replay vs Real Features)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# ======================================================================
#  6. ACCURACY HEATMAP
# ======================================================================

def plot_accuracy_heatmap(tracker, save_path: str):
    """Seaborn heatmap of the accuracy matrix A[t][i]."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    acc_matrix = tracker.accuracy_matrix.copy() * 100  # convert to percentage

    # Mask NaN values
    mask = np.isnan(acc_matrix)

    sns.heatmap(acc_matrix, annot=True, fmt='.1f', mask=mask,
                cmap='YlOrRd', vmin=0, vmax=100,
                xticklabels=[f'Task {i}' for i in range(tracker.num_tasks)],
                yticklabels=[f'After T{t}' for t in range(tracker.num_tasks)],
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Accuracy (%)'},
                ax=ax)

    ax.set_xlabel('Evaluated Task')
    ax.set_ylabel('Learned Up To Task')
    ax.set_title('Task Accuracy Matrix A[t][i]')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# ======================================================================
#  7. COSINE VS FORGETTING CORRELATION
# ======================================================================

def plot_cosine_vs_forgetting(tracker, save_path: str):
    """Scatter plot of cosine similarity vs forgetting per task."""
    _setup_style()

    forgetting = tracker.get_forgetting_per_task()
    cosine_stats = tracker.cosine_stats_per_task

    if not cosine_stats:
        return

    tasks_with_both = []
    cos_vals = []
    forg_vals = []

    for t in range(tracker.num_tasks):
        if t in cosine_stats and t < len(forgetting):
            cos_vals.append(cosine_stats[t].get('mean', 0))
            forg_vals.append(forgetting[t])
            tasks_with_both.append(t)

    if len(tasks_with_both) < 2:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(cos_vals, forg_vals, c=COLORS[0], s=100, edgecolors='black',
               linewidth=0.5, zorder=5)

    # Annotate points
    for t, cx, fy in zip(tasks_with_both, cos_vals, forg_vals):
        ax.annotate(f'T{t}', (cx, fy), textcoords="offset points",
                    xytext=(8, 5), fontsize=9)

    # Trend line
    if len(cos_vals) >= 2:
        z = np.polyfit(cos_vals, forg_vals, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(cos_vals), max(cos_vals), 100)
        ax.plot(x_trend, p(x_trend), '--', color=COLORS[3], linewidth=1.5,
                alpha=0.7, label=f'Trend (slope={z[0]:.3f})')

        # Correlation coefficient
        corr = np.corrcoef(cos_vals, forg_vals)[0, 1]
        ax.set_title(f'Cosine Similarity vs Forgetting (r = {corr:.3f})')
    else:
        ax.set_title('Cosine Similarity vs Forgetting')

    ax.set_xlabel('Mean Cosine Similarity')
    ax.set_ylabel('Forgetting')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# ======================================================================
#  8. COSINE VS ACCURACY CORRELATION
# ======================================================================

def plot_cosine_vs_accuracy(tracker, save_path: str):
    """Scatter plot of cosine similarity vs accuracy per task."""
    _setup_style()

    acc_matrix = tracker.accuracy_matrix
    cosine_stats = tracker.cosine_stats_per_task

    if not cosine_stats:
        return

    cos_vals = []
    acc_vals = []
    tasks_with_both = []

    for t in range(tracker.num_tasks):
        if t in cosine_stats:
            # Use diagonal accuracy (accuracy on task t right after learning it)
            diag_acc = acc_matrix[t, t]
            if not np.isnan(diag_acc):
                cos_vals.append(cosine_stats[t].get('mean', 0))
                acc_vals.append(diag_acc * 100)
                tasks_with_both.append(t)

    if len(tasks_with_both) < 2:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(cos_vals, acc_vals, c=COLORS[2], s=100, edgecolors='black',
               linewidth=0.5, zorder=5)

    for t, cx, ay in zip(tasks_with_both, cos_vals, acc_vals):
        ax.annotate(f'T{t}', (cx, ay), textcoords="offset points",
                    xytext=(8, 5), fontsize=9)

    # Trend line
    if len(cos_vals) >= 2:
        z = np.polyfit(cos_vals, acc_vals, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(cos_vals), max(cos_vals), 100)
        ax.plot(x_trend, p(x_trend), '--', color=COLORS[3], linewidth=1.5,
                alpha=0.7, label=f'Trend (slope={z[0]:.3f})')

        corr = np.corrcoef(cos_vals, acc_vals)[0, 1]
        ax.set_title(f'Cosine Similarity vs Accuracy (r = {corr:.3f})')
    else:
        ax.set_title('Cosine Similarity vs Accuracy')

    ax.set_xlabel('Mean Cosine Similarity')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# ======================================================================
#  ORCHESTRATOR
# ======================================================================

def generate_all_plots(tracker, output_dir: str):
    """
    Generate all research plots and save to output_dir.

    Parameters
    ----------
    tracker : ResearchMetricsTracker
    output_dir : str
        Directory to save plot images.
    """
    os.makedirs(output_dir, exist_ok=True)

    plot_functions = [
        ('accuracy_vs_tasks.png', plot_accuracy_vs_tasks),
        ('forgetting_vs_tasks.png', plot_forgetting_vs_tasks),
        ('communication_cost.png', plot_communication_cost),
        ('bias_comparison.png', plot_bias_comparison),
        ('cosine_distribution.png', plot_cosine_distribution),
        ('accuracy_heatmap.png', plot_accuracy_heatmap),
        ('cosine_vs_forgetting.png', plot_cosine_vs_forgetting),
        ('cosine_vs_accuracy.png', plot_cosine_vs_accuracy),
    ]

    for filename, plot_fn in plot_functions:
        save_path = os.path.join(output_dir, filename)
        try:
            plot_fn(tracker, save_path)
            logger.info("  Plot saved: %s", save_path)
        except Exception as e:
            logger.warning("  Plot failed (%s): %s", filename, str(e))
