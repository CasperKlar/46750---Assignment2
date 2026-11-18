"""Entry point of the Assignment 2 starter package: load a community and summarize it.

    python main.py                                   # base case (16 members), day d1
    python main.py --community my_mix                # your own data/params_community_my_mix.json
    python main.py --day d4                          # another day of the week (and that day's cap)
    python main.py --members                         # also print every member's Assignment 1 summary
    python main.py --demo                            # also run the scenario-generation demonstration (Question 3)

No model is provided: build your Assignment 1 model class on the ``InputData`` of each member, then the community
models of the assignment (R, I, C, DP, ...) on top of it, one function per experiment, writing to ``results/``.
"""

from __future__ import annotations
import pandas as pd
import numpy as np


import argparse
from pathlib import Path
from gurobipy import GRB

from src.config import SENSITIVITY_Q1_BETA, SENSITIVITY_Q1, SENSITIVITY_Q1_BETA_PAIR, SEED_IN, SEED_OUT
from src.data_loader import load_community, list_communities
from src.Q1_models import ModelI, ModelR, ModelC
from src.Q1_results import (collect_model_results, plot_hourly_imports_exports, plot_imports_vs_capacity,
                            print_summary_table, plot_Q1_sensitivity_manual,
                            save_hourly_flows, plot_hourly_flow_difference, plot_duals,
                            member_costs, print_settlement_summary)
from src.Q2_models import (
    build_pricing_problem,
    extract_pricing_result,
    solve_lexicographic_ir,
    community_metrics,
    validate_payments,
    individual_ir_metrics,
)
from src.Q3_models import make_scenarios  
from src.Q3_results import check_scenarios, check_daily_minimum, check_indices  

RESULTS_DIR = Path(__file__).resolve().parent / "results"

def solve_q1_benchmarks(com):
    """Solve Q1 models and return per-member R and I costs,
    along with the centralized Model C benchmark."""

    models = {}

    for name, model_class in (
        ("R", ModelR),
        ("I", ModelI),
        ("C", ModelC),
    ):
        model = model_class(com, verbose=False)
        model.build()
        model.solve()

        if model.m.Status != GRB.OPTIMAL:
            raise RuntimeError(
                f"Model {name} did not solve optimally."
            )

        models[name] = model

    # Reuse the existing Q1_results.py function.
    costs = member_costs(
        models["R"],
        models["I"],
        models["C"]
    )

    # Dictionaries indexed by consumer_id.
    benchmark_R = costs.set_index("member")["J_R"].to_dict()
    benchmark_I = costs.set_index("member")["J_I"].to_dict()

    # Centralized benchmark for efficiency comparisons.
    objective_C = models["C"].m.ObjVal

    return benchmark_R, benchmark_I, objective_C, costs


def run_all_q2_scenarios(com, save_dir):
    """
    Run all Q2 pricing configurations and save results.

    Configurations:
        DP0
        DPBB
        DPBBIR_hard
        DPBBIR_lexicographic
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------
    # 1. Solve Q1 benchmarks once
    # -------------------------------------------
    benchmark_R, benchmark_I, objective_C, costs = (
        solve_q1_benchmarks(com)
    )

    costs.to_csv(
        save_dir / "q1_member_benchmarks.csv",
        index=False
    )

    scenarios = [
        ("DP0", "DP0", False),
        ("DPBB", "DPBB", False),
        ("DPBBIR_hard", "DPBBIR", False),
        ("DPBBIR_lexicographic", "DPBBIR", True),
    ]

    comparison_rows = []
    # Store DPBB's solution to initialize the relaxed IR model.
    dpbb_start = None
    dpbb_result = None

    # -------------------------------------------
    # 2. Solve every Q2 scenario
    # -------------------------------------------
    for label, variant, ir_soft in scenarios:

        print(f"\n{'=' * 60}")
        print(f"Running {label}")
        print(f"{'=' * 60}")

        scenario_dir = save_dir / label
        scenario_dir.mkdir(parents=True, exist_ok=True)

        model, variables = build_pricing_problem(
            com,
            variant=variant,
            benchmark_R=benchmark_R,
            benchmark_I=benchmark_I,
            ir_soft=ir_soft,
            end_E_bat_rule="final",
        )

        model.Params.OutputFlag = 1
        model.Params.TimeLimit = 300
        model.Params.MIPGap = 1e-5
        model.Params.LogFile = str(
            scenario_dir / "gurobi.log"
        )

        # Warm start for lexicographic IR using DPBB.
        if label == "DPBBIR_lexicographic" and dpbb_start is not None:

            model.update()

            # Initialize all matching variables, including
            # the complementarity binary variables.
            for var in model.getVars():
                if var.VarName in dpbb_start:
                    var.Start = dpbb_start[var.VarName]

            # DPBB has no IR slack variables, so initialize
            # the additional slack variables separately.
            for i, slack_var in variables["ir_slack"].items():

                outside_option = min(
                    benchmark_R[i],
                    benchmark_I[i]
                )

                member_cost = dpbb_result.member_costs_DKK[i]

                required_slack = max(
                    0.0,
                    member_cost - outside_option
                )

                # Small margin for numerical feasibility.
                slack_var.Start = required_slack + 1e-4

            print(
                "Warm start initialized using "
                "the DPBB solution."
            )

        row = {
            "Model": label,
            "Status": None,
            "Feasible solution": False,
            "Proven optimal": False,
            "Community cost (DKK)": np.nan,
            "Model C objective (DKK)": objective_C,
            "Efficiency loss (DKK)": np.nan,
            "Grid cost (DKK)": np.nan,
            "Budget residual (DKK)": np.nan,
            "Peak import (kW)": np.nan,
            "Total cap exceedance (kWh)": np.nan,
            "Hours above cap": np.nan,
            "Total IR violation (DKK)": np.nan,
            "Members violating IR": np.nan,
            "Max payment error (DKK)": np.nan,
            "MIP gap": np.nan,
            "Runtime (s)": np.nan,
        }


        lex_info = None
        lex_completed = True
        lex_error = None
        """
        try:
            if ir_soft:
                lex_info = solve_lexicographic_ir(
                    model,
                    variables,
                    ir_tolerance=1e-5,
                    mip_gap=1e-5,
                )
            else:
                model.optimize()

        except RuntimeError as exc:
            lex_completed = False
            lex_error = str(exc)

            print(
                f"{label}: lexicographic optimization "
                f"not completed: {exc}"
            )

        # Preserve solver diagnostics.
        row["Lexicographic completed"] = (
            lex_completed if ir_soft else None
        )
        row["Solver message"] = lex_error

        # For an incomplete lexicographic run, record
        # the Stage 1 incumbent separately.
        if ir_soft and not lex_completed and model.SolCount > 0:

            total_slack = sum(
                var.X
                for var in variables["ir_slack"].values()
            )

            row["Stage 1 incumbent IR slack (DKK)"] = (
                total_slack
            )
            row["Stage 1 best bound (DKK)"] = model.ObjBound

            print(
                f"Stage 1 incumbent slack: "
                f"{total_slack:.3f} DKK"
            )
            print(
                f"Stage 1 best bound: "
                f"{model.ObjBound:.3f} DKK"
            )

        """

        try:
            if ir_soft:
                lex_info = solve_lexicographic_ir(
                    model,
                    variables,
                    ir_tolerance=1e-4,
                    mip_gap=1e-5,
                    allow_provisional=True,
                )
            else:
                model.optimize()

        except RuntimeError as exc:
            lex_completed = False
            lex_error = str(exc)

            print(
                f"{label}: optimization incomplete: "
                f"{exc}"
            )

        row["Lexicographic completed"] = (
            lex_completed if ir_soft else None
        )
        row["Solver message"] = lex_error

        if lex_info is not None:

            row["Stage 1 certified"] = (
                lex_info["stage1_certified"]
            )

            row["Stage 1 best IR slack (DKK)"] = (
                lex_info["stage1_best_slack_DKK"]
            )

            row["Stage 1 best bound (DKK)"] = (
                lex_info["stage1_best_bound_DKK"]
            )

            row["Stage 2 certified"] = (
                lex_info["stage2_certified"]
            )

            row["Provisional"] = (
                lex_info["provisional"]
            )

            row["Stage 1 runtime (s)"] = (
                lex_info["stage1_runtime_s"]
            )

            row["Stage 2 runtime (s)"] = (
                lex_info["stage2_runtime_s"]
            )

        elif ir_soft and model.SolCount > 0:
            # Diagnostic incumbent only.
            # Not a completed Stage 2 result.
            row["Incomplete stage incumbent objective"] = (
                model.ObjVal
            )


        row["Status"] = model.Status

        if lex_info is not None:
            row["Runtime (s)"] = (
                lex_info["stage1_runtime_s"]
                + lex_info["stage2_runtime_s"]
            )
        else:
            row["Runtime (s)"] = model.Runtime

        # Check numerical quality of the best solution.
        if model.SolCount > 0:
            print(f"\n--- Solution quality: {label} ---")
            model.printQuality()

            print(
                "Maximum constraint violation:",
                model.ConstrVio
            )
            print(
                "Maximum bound violation:",
                model.BoundVio
            )

        if model.SolCount > 0 and lex_completed:

            result = extract_pricing_result(
                model, variables
            )

            if label == "DPBB":
                dpbb_start = {
                    var.VarName: var.X
                    for var in model.getVars()
                }
                dpbb_result = result

            metrics = community_metrics(
                result,
                com,
                objective_C=objective_C
            )

            metrics = community_metrics(
                result,
                com,
                objective_C=objective_C
            )

            ir_metrics = individual_ir_metrics(
                result,
                benchmark_R,
                benchmark_I
            )

            payment_error = validate_payments(result)

            # -----------------------------------
            # 3. Community-level metrics
            # -----------------------------------
            row.update({
                "Feasible solution": True,

                "Proven optimal": (
                    lex_info["fully_certified"]
                    if ir_soft
                    else model.Status == GRB.OPTIMAL
                ),
                "Community cost (DKK)":
                    result.objective_DKK,
                "Efficiency loss (DKK)":
                    metrics["efficiency_loss_DKK"],
                "Grid cost (DKK)":
                    result.grid_cost_DKK,
                "Budget residual (DKK)":
                    metrics["budget_residual_DKK"],
                "Peak import (kW)":
                    metrics["peak_import_kW"],
                "Total cap exceedance (kWh)":
                    metrics["cap_exceedance_kWh"],
                "Hours above cap":
                    metrics["hours_above_cap"],
                "Total IR violation (DKK)":
                    ir_metrics["total_ir_violation_DKK"],
                "Members violating IR":
                    ir_metrics["number_violating_ir"],
                "Max payment error (DKK)":
                    payment_error,
                "MIP gap":
                    model.MIPGap,
            })

            if lex_info is not None:
                row["Stage 1 best IR slack (DKK)"] = (
                    lex_info["stage1_best_slack_DKK"]
                )

                row["Stage 2 IR slack (DKK)"] = (
                    lex_info["final_ir_violation_DKK"]
                )

                row["IR slack limit (DKK)"] = (
                    lex_info["ir_slack_limit_DKK"]
                )

            # -----------------------------------
            # 4. Hourly results
            # -----------------------------------
            hourly = pd.DataFrame({
                "hour": list(range(
                    len(result.prices_DKK_per_kWh)
                )),
                "price_DKK_per_kWh":
                    result.prices_DKK_per_kWh,
                "import_kWh":
                    result.import_kWh,
                "export_kWh":
                    result.export_kWh,
                "exceedance_kWh":
                    result.exceedance_kWh,
                "cap_kW":
                    com.contract.capacity_limit_kW,
            })

            hourly.to_csv(
                scenario_dir / "hourly_results.csv",
                index=False
            )

            # -----------------------------------
            # 5. Member-level results
            # -----------------------------------
            member_rows = []

            for member in com.members:
                i = member.consumer_id

                member_rows.append({
                    "member": i,
                    "archetype": member.archetype,
                    "J_R": benchmark_R[i],
                    "J_I": benchmark_I[i],
                    "outside_option": min(
                        benchmark_R[i], benchmark_I[i]
                    ),
                    "J_DP": result.member_costs_DKK[i],
                    "payment_DKK": result.payments_DKK[i],
                    "benefit_DKK":
                        ir_metrics[
                            "benefits_by_member_DKK"
                        ][i],
                    "IR_violation_DKK":
                        ir_metrics[
                            "violations_by_member_DKK"
                        ][i],
                    "IR_slack_DKK":
                        result.ir_slacks_DKK.get(i, 0.0),
                })

            pd.DataFrame(member_rows).to_csv(
                scenario_dir / "member_results.csv",
                index=False
            )

            print(
                f"{label}: "
                f"cost = {result.objective_DKK:.3f} DKK, "
                f"IR violations = "
                f"{ir_metrics['number_violating_ir']}"
            )

        else:
            if ir_soft and not lex_completed:
                print(
                    f"{label}: no completed lexicographic "
                    f"result available. Status = {model.Status}"
                )
            else:
                print(
                    f"{label}: no feasible solution found. "
                    f"Solver status = {model.Status}"
                )

        # ---------------------------------------
        # 6. Save scenario summary
        # ---------------------------------------
        pd.DataFrame([row]).to_csv(
            scenario_dir / "summary.csv",
            index=False
        )

        comparison_rows.append(row)

        # Save comparison incrementally, so
        # completed scenarios are not lost.
        pd.DataFrame(comparison_rows).to_csv(
            save_dir / "q2_comparison.csv",
            index=False
        )

    # -------------------------------------------
    # 7. Final comparison
    # -------------------------------------------
    comparison = pd.DataFrame(comparison_rows)

    print("\n=== Q2 MODEL COMPARISON ===")
    print(comparison.to_string(index=False))

    print(f"\nResults saved to: {save_dir}")

    return comparison



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default="base", help=f"base (shipped) or your own params_community_<name>.json; available: {list_communities()}")
    ap.add_argument("--day", default="d1", help="d1 (base case) ... d7")
    ap.add_argument("--members", action="store_true", help="print the Assignment 1 summary of every member")
    ap.add_argument("--demo", action="store_true", help="run the scenario-generation demonstration")
    #adds our models I,R,C:
    ap.add_argument("--Q1", action="store_true",help="Run all Question 1 models (I, R, C)")
    ap.add_argument("--Q2",choices=["DP0", "DPBB", "DPBBIR"],help="Run Question 2 pricing model")        
    ap.add_argument("--ir-mode",choices=["hard", "lexicographic"],default="hard",help="Individual-rationality treatment for DPBBIR")
    ap.add_argument("--Q2-all",action="store_true",help="Run all Q2 variants and compare results")
    ap.add_argument("--Q2-sensitivity",action="store_true",help="Run the Q2 contract-parameter sensitivity study")
    ap.add_argument("--Q3", choices=["scenarios"], help="Run Question 3 step: generate and check the scenarios")  
    args = ap.parse_args()

    com = load_community(args.community, day=args.day)
    print(com.summary(), "\n")
    
    if args.members:
        for m in com.members:
            print(m.summary(), "\n")
    print("Cap profile (kW):", " ".join(f"{x:.1f}" for x in com.contract.capacity_limit_kW))
    ic = com.contract.individual_capacity_limit_kW
    print("Individual caps in model I (kW), hour 0 / hour 11:",
          "  ".join(f"{cid} ({com.member(cid).archetype}) {ic[cid][0]:.2f}/{ic[cid][11]:.2f}" for cid in ("C1", "C1_PVB", "C4", "C4_PVB") if cid in com.member_ids))
    (RESULTS_DIR / args.community).mkdir(parents=True, exist_ok=True)
    
    if args.demo:
        import runpy
        runpy.run_module("src.scenario_generation", run_name="__main__")

    #Q1
    if args.Q1:
        model_classes = {"I": ModelI, "R": ModelR, "C": ModelC}
        all_results = []
        models = {}


        if SENSITIVITY_Q1:
            save_dir = RESULTS_DIR/ "Q1" / args.community / args.day / SENSITIVITY_Q1_BETA
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = RESULTS_DIR/ "Q1" / args.community / args.day 
            save_dir.mkdir(parents=True, exist_ok=True)

        for name in ("R", "I", "C"):
            model = model_classes[name](com, verbose=True)
            model.build()
            model.solve()

            if model.m.Status != GRB.OPTIMAL:
                print(f"Model {name} did not solve to optimality; status: {model.m.Status}")
                continue

            models[name] = model

            all_results.append(collect_model_results(model, f"Model {name}", com))

        if all_results:
            plot_hourly_imports_exports(all_results, save_dir)
            plot_imports_vs_capacity(all_results, save_dir)
            print_summary_table(all_results, save_dir)
            plot_Q1_sensitivity_manual(RESULTS_DIR / args.community / args.day)
            save_hourly_flows(all_results, save_dir)

            model_c = models.get("C")
            if model_c is not None:   
                plot_duals(model_c.duals, save_dir)
                model_c.duals.to_csv(save_dir / "duals_C.csv")
                print(f"Budget B = {model_c.settlement['B']:.3f} DKK")

            if all(k in models for k in ("R", "I", "C")):
                table = member_costs(models["R"], models["I"], models["C"])
                table.to_csv(save_dir / "member_costs.csv", index=False)
                print_settlement_summary(table, models["R"], models["I"], models["C"],
                                         save_path=save_dir / "settlement_summary.txt")

            # difference plot, only once both beta runs exist
            if SENSITIVITY_Q1:
                day_dir = RESULTS_DIR / args.community / args.day
                tag_a, tag_b = SENSITIVITY_Q1_BETA_PAIR
                dir_a, dir_b = day_dir / tag_a, day_dir / tag_b
                if (dir_a / "hourly_flows.csv").exists() and (dir_b / "hourly_flows.csv").exists():
                    label_a = "β=" + tag_a.split("_")[-1]
                    label_b = "β=" + tag_b.split("_")[-1]
                    plot_hourly_flow_difference(dir_a, dir_b, label_a, label_b, day_dir)

            print(f"Comparison CSV and plots saved to: {save_dir}")

    # Run all Question 2 variants
    if args.Q2_all:
        q2_dir = RESULTS_DIR / args.community / args.day / "Q2"

        run_all_q2_scenarios(
            com,
            save_dir=q2_dir
        )

    # Q2
    if args.Q2:
        print(f"\n--- Running Model {args.Q2} ---")

        # Solve Q1 to obtain the member-level benchmarks.
        benchmark_R, benchmark_I, objective_C, costs = (
            solve_q1_benchmarks(com)
        )

        ir_soft = (
            args.Q2 == "DPBBIR"
            and args.ir_mode == "lexicographic"
        )

        model, variables = build_pricing_problem(
            com,
            variant=args.Q2,
            benchmark_R=benchmark_R,
            benchmark_I=benchmark_I,
            ir_soft=ir_soft,
        )

        model.Params.OutputFlag = 1
        model.Params.TimeLimit = 300
        model.Params.MIPGap = 1e-5

        lex_results = None

        if ir_soft:
            lex_results = solve_lexicographic_ir(
                model,
                variables
            )
        else:
            model.optimize()

        if model.SolCount > 0:
            result = extract_pricing_result(
                model,
                variables
            )

            metrics = community_metrics(
                result,
                com,
                objective_C=objective_C
            )

            ir_metrics = individual_ir_metrics(
                result,
                benchmark_R,
                benchmark_I
            )

            payment_error = validate_payments(result)

            print(f"\n--- Model {args.Q2} Results ---")
            print(f"Solver status: {model.Status}")
            print(f"Community cost: {result.objective_DKK:.3f} DKK")
            print(f"Grid cost: {result.grid_cost_DKK:.3f} DKK")
            print(f"Model C benchmark: {objective_C:.3f} DKK")

            print(
                f"Efficiency loss: "
                f"{metrics['efficiency_loss_DKK']:.3f} DKK"
            )

            print(
                f"Budget residual: "
                f"{metrics['budget_residual_DKK']:.6f} DKK"
            )

            print(
                f"Total IR violation: "
                f"{ir_metrics['total_ir_violation_DKK']:.3f} DKK"
            )

            print(
                f"Members violating IR: "
                f"{ir_metrics['number_violating_ir']}"
            )

            print(
                f"Max payment validation error: "
                f"{payment_error:.6g} DKK"
            )

            print(
                "Internal prices [DKK/kWh]:",
                result.prices_DKK_per_kWh
            )

            if lex_results is not None:
                print("Lexicographic results:", lex_results)

        else:
            print(f"Model {args.Q2} found no feasible solution.")
            print(f"Gurobi status: {model.Status}")

    #Q3
    if args.Q3 == "scenarios":
        print("\n--- Q3: generating and checking scenarios ---")
        save_dir = RESULTS_DIR / "Q3" / args.community / args.day   
        save_dir.mkdir(parents=True, exist_ok=True)          

        #make scenarios
        ins5, ins10, out100 = make_scenarios(save_dir, SEED_IN, SEED_OUT)  

        for name, joint in [("in-sample 5", ins5), ("in-sample 10", ins10), ("out-of-sample 100", out100)]:
            print(f"\nChecks for {name}:")                   
            check_scenarios(joint)

        for name, joint, seed in [("in-sample 5", ins5, SEED_IN),
                          ("in-sample 10", ins10, SEED_IN),
                          ("out-of-sample 100", out100, SEED_OUT)]:
            print(f"\nChecks for {name}:")   
            check_daily_minimum(joint, com)
            check_indices(joint, com, seed)                           

   
    # The model constructors and execution calls will go here
    # once their interfaces are implemented.

if __name__ == "__main__":
    main()

#base is the community configuration
#d1 is the day being simulated.

# Run Q1: python main.py --community base --day d1 --Q1 
# Run Q2: python main.py --community base --day d1 --Q2 DP0
# Run Q3: python main.py --community base --day d1 --Q3 scenarios
