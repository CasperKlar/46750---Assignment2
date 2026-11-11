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

import argparse
from pathlib import Path
from gurobipy import GRB

from src.data_loader import load_community, list_communities
from src.Q1_models import ModelI, ModelR, ModelC
from src.Q1_results import (collect_model_results, plot_hourly_imports_exports, plot_imports_vs_capacity,
                            print_summary_table, plot_Q1_sensitivity_manual,
                            save_hourly_flows, plot_hourly_flow_difference, plot_duals,
                            member_costs, print_settlement_summary)
from src.Q2_models import build_pricing_problem, extract_pricing_result
from src.config import SENSITIVITY_Q1_BETA, SENSITIVITY_Q1, SENSITIVITY_Q1_BETA_PAIR

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default="base", help=f"base (shipped) or your own params_community_<name>.json; available: {list_communities()}")
    ap.add_argument("--day", default="d1", help="d1 (base case) ... d7")
    ap.add_argument("--members", action="store_true", help="print the Assignment 1 summary of every member")
    ap.add_argument("--demo", action="store_true", help="run the scenario-generation demonstration")
    #adds our models I,R,C:
    ap.add_argument("--Q1", action="store_true",help="Run all Question 1 models (I, R, C)")
    ap.add_argument("--Q2",choices=["DP0", "DPBB", "DPBBIR"],help="Run Question 2 pricing model")
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
            save_dir = RESULTS_DIR / args.community / args.day / SENSITIVITY_Q1_BETA
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = RESULTS_DIR / args.community / args.day 
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

    # Q2
    if args.Q2:
        print(f"\n--- Running Model {args.Q2} ---")

        model, variables = build_pricing_problem(
            com,
            variant=args.Q2
        )

        model.Params.OutputFlag = 1
        model.Params.TimeLimit = 300
        model.Params.MIPGap = 0.01

        model.optimize()

        if model.SolCount > 0:
            result = extract_pricing_result(model, variables)

            print(f"\n--- Model {args.Q2} Results ---")
            print(f"Solver status: {model.Status}")
            print(f"Optimality gap: {model.MIPGap:.4%}")
            print(f"Community cost [DKK]: {result.objective_DKK:.3f}")
            print(f"Grid cost [DKK]: {result.grid_cost_DKK:.3f}")
            print(f"Internal prices [DKK/kWh]:")
            print(result.prices_DKK_per_kWh)

        else:
            print(f"Model {args.Q2} found no feasible solution.")
            print(f"Gurobi status: {model.Status}")

   
    # The model constructors and execution calls will go here
    # once their interfaces are implemented.

if __name__ == "__main__":
    main()

#base is the community configuration
#d1 is the day being simulated.

# Run Q1: python main.py --community base --day d1 --Q1 
# Run Q2: python main.py --community base --day d1 --Q2 DP0
