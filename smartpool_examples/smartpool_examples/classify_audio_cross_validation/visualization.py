import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import N_FOLDS


def plot_results(fold_accuracies, output_path="cross_validation_results.png"):
    fig, ax = plt.subplots(figsize=(12, 6))

    model_names = list(fold_accuracies.keys())
    x_positions = []
    x_labels = []
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

    for i, (name, accs) in enumerate(fold_accuracies.items()):
        x = list(range(1, len(accs) + 1))
        x_positions.extend([pos + i * 0.2 for pos in x])
        x_labels.extend([f"Fold {j}" for j in x])
        mean_acc = sum(accs) / len(accs)
        ax.bar([pos + i * 0.2 for pos in x], accs, width=0.18, color=colors[i % len(colors)],
               label=f"{name} (mean: {mean_acc:.4f})")

    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    ax.set_title("5-Fold Cross Validation - Audio Classification")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Results saved to {output_path}")