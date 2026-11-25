
"""Question 2: Grid capacity sensitivity analysis."""

from pathlib import Path
import numpy as np
import pandas as pd

from src.data_loader import load_community
from main import run_all_q2_scenarios, RESULTS_DIR


# ============================================================
# SETTINGS
# ============================================================

COMMUNITY = "base"
DAY = "d1"

CAP_MULTIPLIERS = [0.80, 0.90, 1.00, 1.10, 1.20]

SAVE_DIR = (
    RESULTS_DIR
    / COMMUNITY
    / DAY
    / "Q2"
    / "sensitivity"
    / "grid_capacity"
)


# ============================================================
# RUN SENSITIVITY ANALYSIS
# ============================================================

def main():

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for multiplier in CAP_MULTIPLIERS:

        # Load a fresh community for each experiment
        com = load_community(COMMUNITY, day=DAY)

        # Original grid capacity
        original_cap = np.asarray(
            com.contract.capacity_limit_kW,
            dtype=float
        ).copy()

        # Apply sensitivity multiplier
        new_cap = original_cap * multiplier

        com.contract.capacity_limit_kW = new_cap

        # Separate output folder for each capacity
        tag = f"cap_{round(multiplier * 100):03d}"
        scenario_dir = SAVE_DIR / tag

        print("\n" + "=" * 65)
        print("GRID CAPACITY SENSITIVITY")
        print(f"Multiplier: {multiplier:.2f}")
        print(f"Original cap: {original_cap[0]:.3f} kW")
        print(f"Modified cap: {new_cap[0]:.3f} kW")
        print("=" * 65)

        # Run existing Q2 workflow, unchanged
        comparison = run_all_q2_scenarios(
            com,
            save_dir=scenario_dir
        )

        # Add sensitivity information
        comparison = comparison.copy()
        comparison.insert(0, "Cap multiplier", multiplier)
        comparison.insert(1, "Grid capacity (kW)", new_cap[0])

        all_results.append(comparison)

        # Save consolidated results incrementally
        combined = pd.concat(
            all_results,
            ignore_index=True
        )

        combined.to_csv(
            SAVE_DIR / "sensitivity_summary.csv",
            index=False
        )

    print("\nSensitivity analysis completed.")
    print(f"Results saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
