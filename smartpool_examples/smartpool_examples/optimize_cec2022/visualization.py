import matplotlib.pyplot as plt
import numpy as np
import rich
from rich.table import Table


def print_results_table(results):
    table = Table(title="CEC2022 Optimization Results Summary")
    table.add_column("Function", style="cyan")
    table.add_column("Algorithm", style="magenta")
    table.add_column("Best Mean", style="green")
    table.add_column("Best Std", style="yellow")
    table.add_column("Best Min", style="blue")
    table.add_column("Best Max", style="red")

    for func_name in sorted(results.keys()):
        for algo_name in sorted(results[func_name].keys()):
            fits = [r.best_fitness for r in results[func_name][algo_name]]
            table.add_row(
                func_name,
                algo_name,
                f"{np.mean(fits):.4e}",
                f"{np.std(fits):.4e}",
                f"{np.min(fits):.4e}",
                f"{np.max(fits):.4e}",
            )

    rich.print(table)


def plot_convergence(results):
    n_funcs = len(results)
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    colors = {"DE": "#1f77b4", "PSO": "#ff7f0e"}

    for idx, func_name in enumerate(sorted(results.keys())):
        ax = axes[idx]
        for algo_name in sorted(results[func_name].keys()):
            all_curves = [r.convergence for r in results[func_name][algo_name]]
            min_len = min(len(c) for c in all_curves)
            clipped = [c[:min_len] for c in all_curves]
            mean_curve = np.mean(clipped, axis=0)
            ax.plot(mean_curve, color=colors[algo_name], label=algo_name, linewidth=1.5)

        ax.set_title(func_name, fontsize=10)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fitness")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("cec2022_convergence.png", dpi=300, bbox_inches="tight")
    plt.show()
