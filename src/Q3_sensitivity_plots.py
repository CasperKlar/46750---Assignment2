"""Q3 sensitivity plots. Run from the repository root."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from fractions import Fraction

# SETTINGS
ROOT = Path(__file__).resolve().parents[1] / 'results' / 'Q3' / 'base' / 'd1' / 'sensitivity'
SHOW_PLOTS = True
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13,
                     'legend.fontsize': 11, 'xtick.labelsize': 11,
                     'ytick.labelsize': 11, 'figure.dpi': 120})

# HELPERS
def save_figure(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(directory / f'{name}.png', dpi=300, bbox_inches='tight')


def numeric(df, column, feasible=None):
    values = pd.to_numeric(df.get(column, pd.Series(np.nan, index=df.index)),
                           errors='coerce').to_numpy(dtype=float, copy=True)
    if feasible is not None:
        values[~feasible] = np.nan
    return values


def solver_status(df):
    return tuple(df[c].astype(str).str.lower().eq('true').to_numpy()
                 for c in ['feasible', 'optimal'])


def uncertified_handle():
    return Line2D([], [], color='black', marker='x', linestyle='None',
                  label='Training MIP tolerance not reached')


def grid(ax, axis='both'):
    ax.grid(axis=axis, alpha=.25)
    ax.set_axisbelow(True)


def runtime_bars(ax, df, positions, scale=1):
    feasible, optimal = solver_status(df)
    runtime = numeric(df, 'runtime_s') / scale
    gap = numeric(df, 'mip_gap') * 100
    groups = [(feasible & optimal, 'Within MIP tolerance', 'tab:blue', ''),
              (feasible & ~optimal, 'Feasible, not certified', 'tab:orange', ''),
              (~feasible, 'No feasible solution found', '0.75', '//')]
    for mask, label, color, hatch in groups:
        if mask.any():
            ax.bar(positions[mask], runtime[mask], width=.55, color=color,
                   hatch=hatch, edgecolor='0.35', label=label)
    for x, t, g, f in zip(positions, runtime, gap, feasible):
        if f and np.isfinite(t) and np.isfinite(g):
            ax.annotate(f'{g:.2f}% gap', (x, t), xytext=(0, 5),
                        textcoords='offset points', ha='center', fontsize=10)
    valid = runtime[np.isfinite(runtime)]
    if valid.size:
        ax.set_ylim(0, max(1, valid.max()) * 1.24)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    grid(ax, 'y')


def save_table(table, plots, name):
    plots.mkdir(parents=True, exist_ok=True)
    table.to_csv(plots / f'{name}.csv', index=False, float_format='%.2f')
    print(table.to_string(index=False, float_format=lambda x: f'{x:.2f}'))


# INVESTIGATION 1: SCENARIO COUNT
def plot_scenario_count():
    directory = ROOT / 'scenario_count'
    df = pd.read_csv(directory / 'sensitivity_summary.csv').sort_values('N')
    plots = directory / 'plots'
    n = df['N'].to_numpy(dtype=int)
    feasible, optimal = solver_status(df)
    is_mean = numeric(df, 'is_mean_cost_DKK', feasible)
    oos_mean = numeric(df, 'oos_mean_cost_DKK', feasible)
    valid = feasible & np.isfinite(is_mean) & np.isfinite(oos_mean) & (is_mean != 0)
    table = pd.DataFrame({'N': n[valid], 'IS mean cost (DKK)': is_mean[valid],
                          'OOS mean cost (DKK)': oos_mean[valid],
                          'Difference (%)': 100*(oos_mean[valid]-is_mean[valid])/is_mean[valid]})
    print('\nINVESTIGATION 1: SCENARIO COUNT')
    save_table(table, plots, 'scenario_count_cost_table')
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    positions = np.arange(len(n))
    runtime_bars(ax, df, positions)
    ax.set_xlabel('Number of in-sample scenarios, $N$')
    ax.set_ylabel('Training solver runtime [s]')
    ax.set_xticks(positions, labels=n)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.19), ncol=2, frameon=False)
    fig.tight_layout()
    save_figure(fig, plots, 'scenario_count_runtime')


# INVESTIGATION 2: RISK AVERSION
def plot_risk_aversion():
    directory = ROOT / 'risk_aversion'
    df = pd.read_csv(directory / 'sensitivity_summary.csv').sort_values('weight')
    plots = directory / 'plots'
    w = numeric(df, 'weight')
    feasible, optimal = solver_status(df)
    uncertified = feasible & ~optimal
    is_mean = numeric(df, 'is_mean_cost_DKK', feasible)
    oos_mean = numeric(df, 'oos_mean_cost_DKK', feasible)
    is_cvar = numeric(df, 'is_cvar_DKK', feasible)
    oos_cvar = numeric(df, 'oos_cvar_DKK', feasible)

    # Mean cost and CVaR versus weight
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for ax, (iv, ov, ylabel) in zip(axes, [(is_mean, oos_mean, 'Mean community cost [DKK]'),
                                           (is_cvar, oos_cvar, 'CVaR [DKK]')]):
        ax.plot(w, iv, color='tab:blue', marker='o', linewidth=1.8, label='In-sample')
        ax.plot(w, ov, color='tab:orange', marker='s', linestyle='--', linewidth=1.8, label='Out-of-sample')
        for values in [iv, ov]:
            mask = uncertified & np.isfinite(values)
            ax.scatter(w[mask], values[mask], marker='x', color='black', s=45, zorder=5)
        ax.set_xlabel(r'CVaR weight, $w$')
        ax.set_ylabel(ylabel)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_xlim(-.04, 1.04)
        grid(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    if uncertified.any():
        handles.append(uncertified_handle())
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False)
    fig.tight_layout(rect=[0, .10, 1, 1])
    save_figure(fig, plots, 'risk_aversion_cost_cvar')

    # Empirical risk-return trade-off: marker shapes identify weights.
    # No point annotations or connecting lines: these are not a proven frontier.
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    markers = ['o', 's', '^', 'D', 'P', 'v', 'h', '*']
    weight_handles = []
    for i, weight in enumerate(w):
        if not feasible[i]:
            continue
        marker = markers[i % len(markers)]
        shown = False
        for mean, cvar, color in [(is_mean, is_cvar, 'tab:blue'),
                                  (oos_mean, oos_cvar, 'tab:orange')]:
            if not (np.isfinite(mean[i]) and np.isfinite(cvar[i])):
                continue
            ax.scatter(mean[i], cvar[i], marker=marker, s=115, color=color,
                       edgecolors='white', linewidths=.7, zorder=3)
            if uncertified[i]:
                ax.scatter(mean[i], cvar[i], marker='x', color='black', s=30, zorder=4)
            shown = True
        if shown:
            weight_handles.append(Line2D([], [], marker=marker, color='0.4',
                                         linestyle='None', markersize=8, label=f'$w={weight:.2f}$'))
    series_handles = [Line2D([], [], marker='o', color=c, linestyle='None', label=l)
                      for c, l in [('tab:blue', 'In-sample'), ('tab:orange', 'Out-of-sample')]]
    if uncertified.any():
        series_handles.append(uncertified_handle())
    ax.set_xlabel('Mean community cost [DKK]')
    ax.set_ylabel('CVaR [DKK]')
    ax.margins(x=.12, y=.12)
    grid(ax)
    fig.legend(handles=series_handles, loc='lower center', bbox_to_anchor=(.5, .065),
               ncol=2, frameon=False)
    if weight_handles:
        fig.legend(handles=weight_handles, loc='lower center', bbox_to_anchor=(.5, .005),
                   ncol=min(4, len(weight_handles)), frameon=False)
    fig.tight_layout(rect=[0, .21, 1, 1])
    save_figure(fig, plots, 'risk_aversion_frontier')
    table = pd.DataFrame({'Weight': w, 'IS mean cost': is_mean, 'OOS mean cost': oos_mean,
                          'IS CVaR': is_cvar, 'OOS CVaR': oos_cvar,
                          'MIP gap (%)': numeric(df, 'mip_gap')*100, 'Feasible': feasible})
    print('\nINVESTIGATION 2: RISK AVERSION')
    save_table(table, plots, 'risk_aversion_cost_table')


# INVESTIGATION 3: CHANCE CONSTRAINTS
def plot_chance_constraint():
    directory = ROOT / 'chance_constraint'
    df = pd.read_csv(directory / 'sensitivity_summary.csv').sort_values('epsilon')
    plots = directory / 'plots'
    eps = numeric(df, 'epsilon')
    feasible, optimal = solver_status(df)
    uncertified = feasible & ~optimal
    is_mean = numeric(df, 'is_mean_cost_DKK', feasible)
    oos_mean = numeric(df, 'oos_mean_cost_DKK', feasible)
    iv = numeric(df, 'is_mean_cap_violation', feasible)*100
    ov = numeric(df, 'oos_mean_cap_violation', feasible)*100
    positions = np.arange(len(eps))
    labels = [str(Fraction(float(e)).limit_denominator(100)) for e in eps]

    # Runtime and achieved violation probability; failed solves stay blank.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2))
    runtime_bars(ax1, df, positions, scale=60)
    ax1.set_ylabel('Training solver runtime [min]')
    ax1.legend(loc='upper center', bbox_to_anchor=(.5, -.20), frameon=False)
    ax2.plot(positions, eps*100, color='0.4', linestyle=':', marker='.',
             label='IS violation limit')
    for values, offset, color, label in [(iv, -.16, 'tab:blue', 'In-sample'),
                                          (ov, .16, 'tab:orange', 'Out-of-sample')]:
        valid = feasible & np.isfinite(values)
        ax2.bar(positions[valid]+offset, values[valid], width=.28, color=color, label=label)
        mask = valid & uncertified
        ax2.scatter(positions[mask]+offset, values[mask], marker='x', color='black', s=45, zorder=5)
    ax2.set_ylabel('Grid capacity violation probability [%]')
    ax2.set_ylim(0, 110)
    grid(ax2, 'y')
    handles, _ = ax2.get_legend_handles_labels()
    if uncertified.any():
        handles.append(uncertified_handle())
    ax2.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.20), frameon=False)
    for ax in (ax1, ax2):
        ax.set_xticks(positions, labels=labels)
        ax.set_xlabel(r'Chance constraint tolerance, $\epsilon$')
        ax.set_xlim(-.5, len(eps)-.5)
    fig.tight_layout()
    save_figure(fig, plots, 'chance_constraint_tradeoff')
    table = pd.DataFrame({'Epsilon': eps, 'IS mean cost': is_mean, 'OOS mean cost': oos_mean,
                          'Target violation (%)': eps*100, 'IS violation (%)': iv,
                          'OOS violation (%)': ov,
                          'IS exceedance (kWh)': numeric(df, 'is_mean_cap_exceedance_kWh', feasible),
                          'OOS exceedance (kWh)': numeric(df, 'oos_mean_cap_exceedance_kWh', feasible),
                          'MIP gap (%)': numeric(df, 'mip_gap')*100,
                          'Runtime (s)': numeric(df, 'runtime_s'), 'Feasible': feasible})
    print('\nINVESTIGATION 3: CHANCE CONSTRAINTS')
    save_table(table, plots, 'chance_constraint_cost_table')


# MAIN
def main():
    investigations = {'scenario_count': plot_scenario_count,
                      'risk_aversion': plot_risk_aversion,
                      'chance_constraint': plot_chance_constraint}
    for name, plot_function in investigations.items():
        if (ROOT / name / 'sensitivity_summary.csv').exists():
            plot_function()
        else:
            print(f'Skipping {name}: No sensitivity results available.')
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close('all')


if __name__ == '__main__':
    main()
