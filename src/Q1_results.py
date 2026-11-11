import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


from src.config import plot_fonts,MODEL_C_COLOR, MODEL_I_COLOR, MODEL_R_COLOR

#plotting initialization
plot_fonts()
MODEL_COLORS = {
    "Model R": MODEL_R_COLOR,
    "Model I": MODEL_I_COLOR,
    "Model C": MODEL_C_COLOR,
}

def get_values(variables):
    "helper function to fetch values"
    return {key: variable.X for key, variable in variables.items()}


def collect_model_results(model, model_name, community):
    "combine all model results"
    T = list(model.T)
    I = list(model.I)
    variables = model.var
    capacity = community.contract.capacity_limit_kW

    # Collect total grid imports and exports for each hour.
    if model_name in ("Model R", "Model I"):
        q_imp = get_values(variables["q_imp"])
        q_exp = get_values(variables["q_exp"])
        imports = [sum(q_imp[i, t] for i in I) for t in T]
        exports = [sum(q_exp[i, t] for i in I) for t in T]
    else:
        P_imp = get_values(variables["P_imp"])
        P_exp = get_values(variables["P_exp"])
        imports = [P_imp[t] for t in T]
        exports = [P_exp[t] for t in T]

    # Collect PV generation and load deviation.
    g = get_values(variables["g"])
    d_pos = get_values(variables["d_+"])
    d_neg = get_values(variables["d_-"])
    pv_generation = sum(g[i, t] for i in I for t in T)
    load_deviation = sum(d_pos[i, t] + d_neg[i, t] for i in I for t in T)

    # Collect battery charging and discharging.
    s_ch = get_values(variables["s_ch"])
    s_dis = get_values(variables["s_dis"])
    battery_charge = sum(s_ch[i, t] for i in I for t in T)
    battery_discharge = sum(s_dis[i, t] for i in I for t in T)

    # Compare aggregate imports with the shared DSO capacity profile.
    hourly_excess = [max(0.0, imports[t] - capacity[t]) for t in T]
    hours_above_capacity = sum(imports[t] > capacity[t] + 1e-6 for t in T)
    total_excess = sum(hourly_excess)

    # Calculate the congestion penalty used in each model.
    alpha = community.contract.penalty_DKK_per_kWh
    if model_name == "Model R":
        congestion_penalty = 0.0
        individual_cap_excess = 0.0
    elif model_name == "Model I":
        individual_cap_excess = sum(max(0.0, variables["q_imp"][i, t].X - community.contract.individual_capacity_limit_kW[community.members[i].consumer_id][t]) for i in I for t in T)
        congestion_penalty = alpha * individual_cap_excess
    else:
        congestion_penalty = alpha * total_excess
        individual_cap_excess = np.nan

    return {
        "name": model_name,
        "hours": T,
        "imports": imports,
        "exports": exports,
        "capacity": list(capacity),
        "hourly_excess": hourly_excess,
        "objective": model.m.ObjVal,
        "total_import": sum(imports),
        "total_export": sum(exports),
        "peak_import": max(imports),
        "hours_above_capacity": hours_above_capacity,
        "total_excess_import": total_excess,
        "individual_cap_excess": individual_cap_excess,
        "congestion_penalty": congestion_penalty,
        "load_deviation": load_deviation,
        "pv_generation": pv_generation,
        "battery_charge": battery_charge,
        "battery_discharge": battery_discharge,
    }


def plot_hourly_imports_exports(results, save_dir):
    "Hourly imports and exports for all three models"

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for result in results:
        c = MODEL_COLORS[result["name"]]
        axes[0].plot(result["hours"], result["imports"], marker="o", markersize=3, color=c,  label=result["name"])
        axes[1].plot(result["hours"], result["exports"], marker="o", markersize=3, color=c, label=result["name"])

    axes[0].set_title("Hourly community grid imports")
    axes[0].set_ylabel("Import (kWh)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Hourly community grid exports")
    axes[1].set_xlabel("Hour")
    axes[1].set_ylabel("Export (kWh)")
    axes[1].set_xticks(range(24))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_dir / "hourly_imports_exports.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_imports_vs_capacity(results, save_dir):
    "Compare hourly aggregate imports with the shared DSO capacity"

    plot_fonts()

    fig, ax = plt.subplots(figsize=(12, 5))

    capacity = results[0]["capacity"]
    hours = results[0]["hours"]

    ax.step(hours, capacity, where="mid", linestyle="--", linewidth=2, color="red", label="DSO capacity limit")

    for result in results:
        ax.plot(hours, result["imports"], marker="o", markersize=3, color=MODEL_COLORS[result["name"]], label=result["name"])

    ax.set_title("Hourly community imports versus DSO capacity")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Import / capacity (kWh per hourly interval)")
    ax.set_xticks(range(24))
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_dir / "import_vs_capacity.png", dpi=300, bbox_inches="tight")
    plt.close()


def print_summary_table(results, save_dir):
    "Print the comparison table for the three models."
    
    rows = []
    for result in results:
        rows.append({
            "Model": result["name"],
            "Objective (DKK)": result["objective"],
            "Total imports (kWh)": result["total_import"],
            "Total exports (kWh)": result["total_export"],
            "Peak import (kWh/hour)": result["peak_import"],
            "Hours above shared cap": result["hours_above_capacity"],
            "Excess above shared cap (kWh)": result["total_excess_import"],
            "Individual cap excess (kWh)": result["individual_cap_excess"],
            "Congestion penalty (DKK)": result["congestion_penalty"],
            "Load deviation (kWh)": result["load_deviation"],
            "PV generation (kWh)": result["pv_generation"],
            "Battery charge (kWh)": result["battery_charge"],
            "Battery discharge (kWh)": result["battery_discharge"],
        })

    summary = pd.DataFrame(rows).set_index("Model")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(save_dir / f"model_comparison.csv", index=True)

    return summary


def plot_Q1_sensitivity_manual(save_dir):
    "plots the results for the sensitivity, which are manually inputtet per the results generated"

    plot_fonts()
    
    # Beta values
    beta_values = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

    # 1. OBJECTIVE VALUES (DKK)
    profit_R = [412.772, 412.772, 412.772, 412.772, 412.772, 412.772, 412.772]
    profit_I = [502.148, 497.639, 493.118, 488.566, 483.998, 479.402, 474.806]
    profit_C = [391.740, 386.797, 381.785, 376.607, 371.429, 366.251, 361.038]

    plt.figure(figsize=(8, 5))

    plt.plot(beta_values, profit_R, marker='v', color=MODEL_R_COLOR, label='Base (R)')
    plt.plot(beta_values, profit_I, marker='o', color=MODEL_I_COLOR, label='Individual contract (I)')
    plt.plot(beta_values, profit_C, marker='s', color=MODEL_C_COLOR, label='Community contract (C)')

    plt.xlabel(r'$\beta^{DSO}$')
    plt.ylabel('Objective (DKK)')
    plt.title('Objective value as a Function of DSO Discount')
    plt.xticks(beta_values)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / "ProfitQ1.png", dpi=300, bbox_inches="tight")

    # 2. Load deviation (kWh)
    #load_dev_R = [14.780, 14.780, 14.780, 14.780, 14.780, 14.780, 14.780]
    load_dev_I = [117.199, 117.199, 117.199, 117.746, 117.808, 118.250, 119.050]
    load_dev_C = [111.260, 111.260, 114.213, 114.213, 114.213, 114.213, 114.213]

    plt.figure(figsize=(8, 5))

    #plt.plot(beta_values, load_dev_R, marker='v', label='Base (R)')
    plt.plot(beta_values, load_dev_I, marker='o', color=MODEL_I_COLOR, label='Individual contract (I)')
    plt.plot(beta_values, load_dev_C, marker='s', color=MODEL_C_COLOR, label='Community contract (C)')

    plt.xlabel(r'$\beta^{DSO}$')
    plt.ylabel('Load Deviation (kWh)')
    plt.title('Load Deviation as a Function of DSO Discount')
    plt.xticks(beta_values)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / "LoadDevSQ1.png", dpi=300, bbox_inches="tight")


def save_hourly_flows(results, save_dir):
    """Save hourly imports/exports of all models to <save_dir>/hourly_flows.csv"""
    rows = [
        {"model": r["name"], "hour": t, "import": imp, "export": exp}
        for r in results
        for t, imp, exp in zip(r["hours"], r["imports"], r["exports"])
    ]
    pd.DataFrame(rows).to_csv(Path(save_dir) / "hourly_flows.csv", index=False)


def plot_hourly_flow_difference(dir_a, dir_b, label_a, label_b, save_dir):
    """Plot hourly import/export difference (b - a) per model from two saved CSVs."""
    a = pd.read_csv(Path(dir_a) / "hourly_flows.csv")
    b = pd.read_csv(Path(dir_b) / "hourly_flows.csv")

    df = a.merge(b, on=["model", "hour"], suffixes=("_a", "_b"))
    df["d_import"] = df["import_b"] - df["import_a"]
    df["d_export"] = df["export_b"] - df["export_a"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for model, g in df.groupby("model", sort=False):
        c = MODEL_COLORS[model]
        axes[0].plot(g["hour"], g["d_import"], marker="o", markersize=3, color=c, label=model)
        axes[1].plot(g["hour"], g["d_export"], marker="o", markersize=3, color=c, label=model)

    for ax in axes:
        ax.axhline(0, color="black", lw=0.8)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_title(f"Change in hourly grid imports (({label_b}) − ({label_a}))")
    axes[0].set_ylabel("Δ Import (kWh)")
    axes[1].set_title(f"Change in hourly grid exports (({label_b}) − ({label_a}))")
    axes[1].set_xlabel("Hour")
    axes[1].set_ylabel("Δ Export (kWh)")
    axes[1].set_xticks(range(24))

    plt.tight_layout()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / "hourly_flows_difference.png", dpi=300, bbox_inches="tight")
    plt.close()