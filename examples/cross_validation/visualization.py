import matplotlib.pyplot as plt
import numpy as np
import rich
from rich.table import Table
from typing import Dict, Any


def plot_results(model_results:Dict[str, Any], stats:Dict[str, Any]):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    plt.rcParams["font.family"] = ["Microsoft YaHei", "Arial", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    
    x_pos = np.arange(len(model_results))
    bar_width = 0.35
    
    for fold_idx in range(5):
        fold_accuracies = []
        for module_result in model_results.values():
            if fold_idx < len(module_result):
                fold_accuracies.append(module_result[fold_idx])
            else:
                fold_accuracies.append(0)
        
        ax.bar(x_pos + fold_idx * bar_width/5, fold_accuracies, 
               bar_width/5, alpha=0.7, label=f'Fold {fold_idx+1}')
    
    means = [stat['mean'] for stat in stats.values()]
    stds = [stat['std'] for stat in stats.values()]
    
    ax.errorbar(x_pos, means, yerr=stds, fmt='o', color='black', 
                capsize=5, capthick=2, elinewidth=2, markersize=8,
                label='Mean ± Std Dev')
    
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.annotate(f'{mean:.4f}\n±{std:.4f}', 
                   xy=(x_pos[i], mean), 
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center', va='bottom',
                   fontsize=9)
    
    ax.set_xlabel('Model Type', fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title('Handwritten Digit Recognition Models: 5-Fold Cross-Validation Comparison', fontsize=14, pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_results.keys())
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.8, 1.0)
    
    plt.tight_layout()
    plt.savefig('cross_validation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_results_table(stats:Dict[str, Any]):
    table = Table(title="5-Fold Cross-Validation Results Summary")
    table.add_column("Model", style="cyan")
    table.add_column("Mean Accuracy", style="magenta")
    table.add_column("Std Deviation", style="green")
    table.add_column("Min", style="yellow")
    table.add_column("Max", style="blue")
    
    for model_name, stat in stats.items():
        stat = stats[model_name]
        table.add_row(
            model_name,
            f"{stat['mean']:.4f}",
            f"{stat['std']:.4f}",
            f"{stat['min']:.4f}",
            f"{stat['max']:.4f}"
        )
    
    rich.print(table)