"""Q3 sensitivity plots. Run from the repository root."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ============================================================
# SETTINGS
# ============================================================
ROOT = Path(__file__).resolve().parents[1] / 'results' / 'Q3' / 'base' / 'd1' / 'sensitivity'
SHOW_PLOTS = True

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 120,
})


def save_figure(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(directory / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(directory / f'{name}.png', dpi=300, bbox_inches='tight')


# ============================================================
# INVESTIGATION 1: SCENARIO COUNT
# ============================================================
def plot_scenario_count():
    directory = ROOT / 'scenario_count'
    df = pd.read_csv(directory / 'sensitivity_summary.csv').sort_values('N')
    plots = directory / 'plots'
    n = df['N'].to_numpy(dtype=int)
    feasible = df['feasible'].astype(str).str.lower().eq('true').to_numpy()
    optimal = df['optimal'].astype(str).str.lower().eq('true').to_numpy()

    def numeric(column):
        return pd.to_numeric(df.get(column, pd.Series(np.nan, index=df.index)), errors='coerce').to_numpy(dtype=float)

    # Table 1: IS and OOS mean community cost (feasible cases only)
    plots.mkdir(parents=True, exist_ok=True)
    table = df.loc[feasible, ['N', 'is_mean_cost_DKK', 'oos_mean_cost_DKK']].copy()
    table['Difference (%)'] = 100 * (table['oos_mean_cost_DKK'] - table['is_mean_cost_DKK']) / table['is_mean_cost_DKK']
    table.columns = ['N', 'IS mean cost (DKK)', 'OOS mean cost (DKK)', 'Difference (%)']
    table.to_csv(plots / 'scenario_count_cost_table.csv', index=False, float_format='%.2f')
    print(table.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

    # Plot 2: Training runtime and achieved MIP gap
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    positions = np.arange(len(n))  # Equally spaced categorical bars
    runtime = numeric('runtime_s')
    gap = numeric('mip_gap') * 100
    groups = [
        (feasible & optimal, 'Within MIP tolerance', 'tab:blue', ''),
        (feasible & ~optimal, 'Feasible, not certified', 'tab:orange', ''),
        (~feasible, 'No feasible solution found', '0.75', '//'),
    ]
    for mask, label, color, hatch in groups:
        if mask.any():
            ax.bar(positions[mask], runtime[mask], width=0.55, color=color, hatch=hatch,
                   edgecolor='0.35', label=label)
    for x, t, g, f in zip(positions, runtime, gap, feasible):
        if f and np.isfinite(t) and np.isfinite(g):
            label = f'{g:.2f}% gap'
            ax.annotate(label, (x, t), xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=10)
    ax.set_xlabel('Number of in-sample scenarios, $N$')
    ax.set_ylabel('Training solver runtime [s]')
    ax.set_xticks(positions, labels=n)
    ax.set_ylim(0, max(1, np.nanmax(runtime)) * 1.24)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis='y', alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.19), ncol=2, frameon=False)
    save_figure(fig, plots, 'scenario_count_runtime')
    print(f'Plots saved to {plots}')


def main():
    plot_scenario_count()
    # Future investigations:
    # plot_risk_aversion()
    # plot_chance_constraint()
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close('all')


if __name__ == '__main__':
    main()
