
"""
Question 2 — Grid capacity sensitivity plots.

Reads previously saved CSV results.
Does not solve any optimization models.

Run from the project root:
    python -m src.Q2_sensitivity_plots
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

COMMUNITY = "base"
DAY = "d1"

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    ROOT / "results" / COMMUNITY / DAY / "Q2"
    / "sensitivity" / "grid_capacity_v1"
)

PLOT_DIR = DATA_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "DP0",
    "DPBB",
    "DPBBIR_hard",
    "DPBBIR_lexicographic",
]

LABELS = {
    "DP0": "DP0",
    "DPBB": "DP-BB",
    "DPBBIR_hard": "DP-BB-IR-hard",
    "DPBBIR_lexicographic": "DP-BB-IR",
}

COLORS = {
    "DP0": "tab:blue",
    "DPBB": "tab:orange",
    "DPBBIR_hard": "tab:red",
    "DPBBIR_lexicographic": "tab:green",
}

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results():

    path = DATA_DIR / "sensitivity_summary.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing sensitivity summary:\n{path}"
        )

    df = pd.read_csv(path)

    df = df.sort_values(
        ["Cap multiplier", "Model"]
    ).reset_index(drop=True)

    # Robust conversion of saved feasibility flags
    df["Feasible solution"] = (
        df["Feasible solution"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    df["Proven optimal"] = (
        df["Proven optimal"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    return df


def model_data(df, model):

    data = df[
        (df["Model"] == model)
        & (df["Feasible solution"])
    ].copy()

    return data.sort_values("Cap multiplier")


def save_plot(fig, filename):

    fig.tight_layout()

    fig.savefig(
        PLOT_DIR / filename,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {filename}")


def add_model_lines(ax, df, metric):

    for model in MODELS:

        data = model_data(df, model)

        if data.empty:
            continue

        ax.plot(
            data["Cap multiplier"] * 100,
            data[metric],
            marker="o",
            linewidth=2,
            markersize=6,
            color=COLORS[model],
            label=LABELS[model],
        )

        # Mark solutions without certified optimality
        provisional = data[
            ~data["Proven optimal"]
        ]

        if not provisional.empty:
            ax.scatter(
                provisional["Cap multiplier"] * 100,
                provisional[metric],
                marker="x",
                s=75,
                linewidths=1.8,
                color="black",
                zorder=5,
            )

    ax.grid(alpha=0.25)
    ax.set_xlabel("Grid capacity [% of baseline]")


# ============================================================
# FIGURE 1 — TOTAL COMMUNITY COST
# ============================================================

def plot_community_cost(df):

    fig, ax = plt.subplots(figsize=(9, 5.5))

    add_model_lines(
        ax, df, "Community cost (DKK)"
    )

    # Centralized benchmark for each capacity
    benchmark = (
        df.groupby("Cap multiplier", as_index=False)
        ["Model C objective (DKK)"]
        .first()
        .sort_values("Cap multiplier")
    )

    ax.plot(
        benchmark["Cap multiplier"] * 100,
        benchmark["Model C objective (DKK)"],
        "k--",
        linewidth=2,
        label="Model C",
    )

    ax.set_ylabel("Community cost [DKK]")
    ax.legend()

    save_plot(fig, "Q2_community_cost.png")


# ============================================================
# FIGURE 2 — EFFICIENCY LOSS
# ============================================================

def plot_efficiency_loss(df):

    fig, ax = plt.subplots(figsize=(9, 5.5))

    add_model_lines(
        ax, df, "Efficiency loss (DKK)"
    )

    ax.axhline(
        0,
        color="black",
        linestyle="--",
        alpha=0.5,
    )

    ax.set_ylabel("Efficiency loss [DKK]")
    ax.legend()

    save_plot(fig, "Q2_efficiency_loss.png")


# ============================================================
# FIGURE 3 — TOTAL IR VIOLATION
# ============================================================

def plot_total_ir_violation(df):

    fig, ax = plt.subplots(figsize=(9, 5.5))

    add_model_lines(
        ax, df, "Total IR violation (DKK)"
    )

    ax.set_ylabel("Total IR violation [DKK]")
    ax.legend()

    save_plot(fig, "Q2_total_IR_violation.png")


# ============================================================
# FIGURE 4 — NUMBER OF MEMBERS VIOLATING IR
# ============================================================

def plot_members_violating_ir(df):

    fig, ax = plt.subplots(figsize=(9, 5.5))

    add_model_lines(
        ax, df, "Members violating IR"
    )

    ax.set_ylabel("Members violating IR [-]")

    max_members = df["Members violating IR"].max()

    if pd.notna(max_members):
        ax.set_yticks(
            np.arange(0, int(max_members) + 2, 2)
        )

    ax.legend()

    save_plot(fig, "Q2_members_violating_IR.png")


# ============================================================
# FIGURE 5 — COMBINED BLOCKING-MEMBER HEATMAPS
# ============================================================

def plot_blocking_members(df):

    capacities = sorted(
        df["Cap multiplier"].unique()
    )

    heatmap_models = [
        ("DP0", "DP0"),
        ("DPBB", "DP-BB"),
        ("DPBBIR_lexicographic", "DP-BB-IR"),
    ]

    matrices = {}

    # --------------------------------------------------------
    # LOAD MEMBER RESULTS FOR EACH MODEL
    # --------------------------------------------------------

    for model, title in heatmap_models:

        records = []

        for multiplier in capacities:

            tag = f"cap_{round(multiplier * 100):03d}"

            path = (
                DATA_DIR / tag / model
                / "member_results.csv"
            )

            if not path.exists():
                continue

            members = pd.read_csv(path)

            members["Cap multiplier"] = multiplier

            records.append(members)

        if not records:
            print(f"No member results found for {model}")
            continue

        members = pd.concat(
            records, ignore_index=True
        )

        matrix = members.pivot_table(
            index="member",
            columns="Cap multiplier",
            values="IR_violation_DKK",
            aggfunc="first",
        )

        matrix = matrix.reindex(
            columns=capacities
        )

        matrices[model] = matrix

    if not matrices:
        print("No blocking-member data available.")
        return

    # --------------------------------------------------------
    # COMMON MEMBER ORDERING
    # --------------------------------------------------------

    all_members = sorted(
        set().union(
            *[set(m.index) for m in matrices.values()]
        )
    )

    # Maximum violation across all models and capacities
    max_violation = pd.concat(
        [
            matrix.reindex(index=all_members).max(axis=1)
            for matrix in matrices.values()
        ],
        axis=1
    ).max(axis=1)

    member_order = (
        max_violation
        .sort_values(ascending=False)
        .index
    )

    # Apply identical ordering to every heatmap
    for model in matrices:

        matrices[model] = matrices[model].reindex(
            index=member_order,
            columns=capacities,
        )

    # --------------------------------------------------------
    # SHARED COLOR SCALE
    # --------------------------------------------------------

    vmax = max(
        np.nanmax(matrix.to_numpy(dtype=float))
        for matrix in matrices.values()
    )

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("lightgray")

    # --------------------------------------------------------
    # CREATE COMBINED FIGURE
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, max(6, len(member_order) * 0.40)),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.87,
        bottom=0.12,
        top=0.90,
        wspace=0.08,
    )

    im = None

    for i, (model, title) in enumerate(heatmap_models):

        ax = axes[i]

        if model not in matrices:
            ax.set_visible(False)
            continue

        matrix = matrices[model]

        values = np.ma.masked_invalid(
            matrix.to_numpy(dtype=float)
        )

        im = ax.imshow(
            values,
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
        )

        ax.set_title(
            title,
            fontsize=16,
            fontweight="bold",
            pad=12,
        )

        ax.set_xticks(
            np.arange(len(capacities))
        )

        ax.set_xticklabels([
            f"{x * 100:.0f}%"
            for x in capacities
        ])

        ax.set_yticks(
            np.arange(len(member_order))
        )

        if i == 0:

            ax.set_yticklabels(member_order)

            ax.set_ylabel(
                "Community member",
                fontsize=15,
            )

            ax.tick_params(
                axis="y",
                labelleft=True,
            )

        else:

            ax.tick_params(
                axis="y",
                left=False,
                labelleft=False,
            )

        ax.tick_params(
            axis="both",
            labelsize=12,
        )

    # --------------------------------------------------------
    # SHARED X-AXIS LABEL
    # --------------------------------------------------------

    fig.supxlabel(
        "Grid capacity [% of baseline]",
        fontsize=15,
        y=0.035,
    )

    # --------------------------------------------------------
    # ONE SHARED COLORBAR
    # --------------------------------------------------------

    if im is not None:

        cax = fig.add_axes([
            0.89, 0.12, 0.022, 0.78
        ])

        cbar = fig.colorbar(
            im,
            cax=cax,
        )

        cbar.set_label(
            "IR violation [DKK]",
            fontsize=15,
        )

        cbar.ax.tick_params(
            labelsize=12
        )

    # Save directly to preserve manual layout
    filename = "Q2_blocking_members_combined.png"

    fig.savefig(
        PLOT_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {filename}")


# ============================================================
# FIGURE 6 — PEAK GRID IMPORT
# ============================================================

def plot_peak_import(df):

    fig, ax = plt.subplots(figsize=(9, 5.5))

    add_model_lines(
        ax, df, "Peak import (kW)"
    )

    ax.set_ylabel("Peak import [kW]")
    ax.legend()

    save_plot(fig, "Q2_peak_import.png")


# ============================================================
# FIGURE 7 — TOTAL CAP EXCEEDANCE
# ============================================================

def plot_cap_exceedance(df):

    fig, ax = plt.subplots(figsize=(9, 5.5))

    add_model_lines(
        ax, df, "Total cap exceedance (kWh)"
    )

    ax.axhline(
        0,
        color="black",
        linestyle="--",
        alpha=0.5,
    )

    ax.set_ylabel("Total cap exceedance [kWh]")
    ax.legend()

    save_plot(fig, "Q2_cap_exceedance.png")


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_results()

    print("\nAvailable scenarios:")
    print(
        df[
            [
                "Cap multiplier",
                "Model",
                "Feasible solution",
                "Proven optimal",
            ]
        ].to_string(index=False)
    )

    plot_community_cost(df)
    plot_efficiency_loss(df)
    plot_total_ir_violation(df)
    plot_members_violating_ir(df)
    plot_blocking_members(df)
    plot_peak_import(df)
    plot_cap_exceedance(df)

    print(f"\nAll plots saved to:\n{PLOT_DIR}")


if __name__ == "__main__":
    main()
