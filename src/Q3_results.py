import numpy as np
import pandas as pd
from src import scenario_generation as sg
from src.Q2_models import (build_pricing_problem, extract_pricing_result,
                           community_metrics, individual_ir_metrics) 
from src.Q3_models import scenario_community, cvar_of 



def check_scenarios(joint, lo=0.8, hi=1.2):
    """Check the drawn theta values and the scenario probabilities.

    Args:
        joint: list of joint scenarios (dicts with keys "theta" and "probability").
        lo: float, lower bound of theta (dimensionless), default 0.8.
        hi: float, upper bound of theta (dimensionless), default 1.2.

    Returns:
        np.ndarray: theta values, shape (number of scenarios, number of members),
        dimensionless. Prints the bounds check, min/max, mean and probability check.
    """
    ids = list(joint[0]["theta"]["profile"].keys())     # the 16 member IDs, taken from the first scenario
    thetas = np.array([[sc["theta"]["profile"][c] for c in ids]
                       for sc in joint])                # table: one row per scenario, one column per member
    probs = np.array([sc["probability"] for sc in joint])  # probability of each joint scenario

    print("theta inside bounds:", thetas.min() >= lo and thetas.max() <= hi)   # must be True
    print("theta min / max:", round(thetas.min(), 3), round(thetas.max(), 3))  # see how close to the bounds
    print("mean of theta:", round(thetas.mean(), 3), "(target 1)")             # roughly 1 (very close for 100)
    print("probabilities sum to 1:", np.isclose(probs.sum(), 1.0))             # must be True
    return thetas

def check_daily_minimum(joint, com):
    """Check that every scenario's daily minimum energy equals its reference energy.

    Args:
        joint: list of joint scenarios (dicts with keys "id", "theta", "members", ...),
            as returned by sg.draw_preference_scenarios.
        com: CommunityData from load_community; its raw appliance catalogue provides
            the unscaled load profile and max_load_KWh_per_hour.

    Returns:
        bool: True if the stored daily minimum matches the recomputed value
        (relative tolerance 1e-6) for every member in every scenario.
        Prints a line for each mismatch and a summary line [kWh/day].
    """

    # unscaled loads of the catalogue (day d1 profiles): the independent reference for the check
    catalogue = com.raw["appliance_params"]["loads"]
    ok = True
    for sc in joint:                                         # every joint scenario
        for cid, appliances in sc["members"].items():        # every member in that scenario
            load_id = next(a for a in appliances if a.startswith("FL_"))   # the member's load entry
            stored = appliances[load_id]["min_total_energy_per_day_kWh"]   # daily minimum the scenario holds
            base = next(l for l in catalogue if l["load_id"] == load_id)   # the same load, unscaled
            rho = np.array(base["reference_load_capacity_factor"])         # unscaled hourly factors
            theta = sc["theta"]["profile"][cid]                            # this member's theta
            expected = base["max_load_KWh_per_hour"] * np.sum(rho ** theta)  # E_min = 3 * sum(rho^theta)
            if not np.isclose(stored, expected, rtol=1e-6):
                ok = False
                print("MISMATCH scenario", sc["id"], cid, stored, expected)
    print("daily minimum equals reference energy in every scenario:", ok)
    return ok

def check_indices(joint, com, seed):
    """Check that the stored indices recover the draws of theta.

    Args:
        joint: list of joint scenarios under test; len(joint) is the number of draws
            per member.
        com: CommunityData from load_community; used for the list of member IDs.
        seed: int, the seed that generated `joint` (SEED_IN or SEED_OUT).
            It must match, otherwise a different sequence is redrawn.

    Returns:
        bool: True if the redrawn theta equals the scenario's theta for every
        member in every scenario. Prints a line for each mismatch and a
        summary line.
    """
    rules = sg.load_json("preference_uncertainty_params.json")   # same rules file as in make_scenarios
    ids = [c["consumer_id"] for c in com.raw["params"]["consumers"]]   # the 16 member IDs
    spec = sg.rules_to_spec(rules, ids)                          # rebuild the draw rules from the file
    # same number of draws as the set under test; other hyper-parameters stay as in the file
    sg.set_rule(spec, "profile", None, n_scenarios=len(joint))
    scaling = sg.draw_scaling_scenarios(spec, seed=seed)         # redraw the independent thetas, SAME seed
    ok = True
    for sc in joint:
        for cid in ids:
            k = sc["index"][("profile", cid)]                    # which independent draw this scenario took
            if not np.isclose(scaling["profile"][cid][k], sc["theta"]["profile"][cid]):
                ok = False                                       # index does not point to the theta used
                print("INDEX MISMATCH scenario", sc["id"], cid)
    print("indices recover the draws:", ok)
    return ok

def evaluate_prices(com, x, scenarios, benchmark_R, benchmark_I, tol=1e-4):
    """Evaluate a fixed price vector with the Q2 pricing model DP on a set of scenarios.

    Per scenario:
      1. Build the scenario community (scenario_community).
      2. Solve the Q2 pricing model with the price fixed (price_lower = price_upper = x).
         Only the members' responses remain free. Among equally good member
         responses, the solver picks the one with the lowest manager cost, which
         is the optimistic assumption of Question 2.
      3. Compute the budget residual, the cap exceedance and the IR violations
         against the outside options.

    Args:
        com: CommunityData, the base community (theta = 1).
        x: array-like, shape (24,), posted internal prices [DKK/kWh].
        scenarios: list of joint scenarios (dicts), e.g. the in-sample or
            out-of-sample set from make_scenarios.
        benchmark_R: dict {member_id: cost [DKK]}, cost without contract (Q1).
        benchmark_I: dict {member_id: cost [DKK]}, cost with an individual
            contract (Q1). A member's outside option is min(benchmark_R,
            benchmark_I). The same values are used in every scenario (computed
            once at theta = 1, as in Q2).
        tol: float, [DKK]. A budget residual below -tol counts as a violation,
            which absorbs solver round-off.

    Returns:
        pd.DataFrame: one row per scenario with columns
            scenario              scenario id
            community_cost        grid cost + members' physical cost [DKK]
            budget_residual       payments collected - grid cost [DKK] (negative = deficit)
            deficit               max(0, -budget_residual) [DKK]
            cap_exceedance_kWh    total import above the cap over the day [kWh]
            hours_above_cap       number of hours with import above the cap
            n_ir_violations       number of members worse off than their outside option
            total_ir_violation    sum of the members' IR violations [DKK]
            budget_violated       True if the deficit is larger than tol
    """
    x = np.asarray(x, dtype=float)
    rows = []

    for sc in scenarios:
        # community with this scenario's profiles, bounds and daily minima
        c = scenario_community(com, sc)

        # Q2 pricing model with the price fixed: lower bound = upper bound = x,
        # so the manager cannot change the price and only member responses are solved
        model, v = build_pricing_problem(c, variant="DP0",
                                         price_lower=x, price_upper=x)
        model.Params.OutputFlag = 0      # silent solver
        model.Params.TimeLimit = 60      # seconds per scenario
        model.Params.MIPGap = 1e-6       # tight gap, so ties are resolved reliably
        model.optimize()

        # extract flows, payments and member costs from the solved model
        res = extract_pricing_result(model, v)
        met = community_metrics(res, c)                            # budget, cap exceedance, ...
        ir = individual_ir_metrics(res, benchmark_R, benchmark_I)  # benefit and IR violation per member

        rows.append({
            "scenario": sc["id"],
            "community_cost": res.objective_DKK,
            "budget_residual": met["budget_residual_DKK"],         # payments - grid cost
            "deficit": max(0.0, -met["budget_residual_DKK"]),      # only the negative part
            "cap_exceedance_kWh": met["cap_exceedance_kWh"],
            "hours_above_cap": met["hours_above_cap"],
            "n_ir_violations": ir["number_violating_ir"],
            "total_ir_violation": ir["total_ir_violation_DKK"],
            "budget_violated": met["budget_residual_DKK"] < -tol,  # deficit beyond round-off
        })

    return pd.DataFrame(rows)

def summarize(df):
    """Summarize the output of evaluate_prices over a scenario set.

    Args:
        df: pd.DataFrame returned by evaluate_prices.

    Returns:
        dict: frequency and magnitude of the three requirement violations.
    """
    return {
        # budget: residual = payments collected - grid cost (positive = surplus, negative = deficit)
        "mean budget residual [DKK]": df["budget_residual"].mean(),
        "min budget residual [DKK]": df["budget_residual"].min(),
        "max budget residual [DKK]": df["budget_residual"].max(),
        "P(budget surplus)": (df["budget_residual"] > 1e-4).mean(),
        "P(budget deficit)": df["budget_violated"].mean(),
        "mean deficit [DKK]": df["deficit"].mean(),
        "max deficit [DKK]": df["deficit"].max(),
        # cap
        "mean cap exceedance [kWh]": df["cap_exceedance_kWh"].mean(),
        "mean hours above cap": df["hours_above_cap"].mean(),
        # individual rationality
        "P(at least one IR violation)": (df["n_ir_violations"] > 0).mean(),
        "mean members violating IR": df["n_ir_violations"].mean(),
        "mean total IR violation [DKK]": df["total_ir_violation"].mean(),
        # cost
        "mean community cost [DKK]": df["community_cost"].mean(),
    }


def extract_stochastic_result(model, v):
    """Extract prices and per-scenario results after optimize() found a feasible solution.

    Returns:
        dict with
            prices            posted prices x [DKK/kWh], shape (24,)
            objective         value of the minimized objective [DKK]
            expected_cost     E[C_s] [DKK]
            cvar_cost         CVaR_alpha of C_s [DKK] (computed from the scenario costs)
            cost_s            C_s per scenario [DKK]
            grid_cost_s       grid cost per scenario [DKK]
            budget_s          B_s per scenario [DKK] (payments - grid cost)
            exceedance_s      cap exceedance per scenario [kWh]
            import_s          community import per scenario, shape (S, 24) [kWh]
            member_cost_s     list over scenarios of {member: D_{i,s}} [DKK]
            ir_slacks         {member: slack [DKK]} (empty if hard IR / no IR)
    """
    if model.SolCount == 0:
        raise RuntimeError(f'No solution found. Gurobi status: {model.Status}')
    S, hours, IDs = v['S'], v['hours'], v['ids']
    cost_s = np.array([v['cost'][s].getValue() for s in range(S)])
    return dict(
        prices=np.array([v['x'][t].X for t in hours]),
        objective=model.ObjVal,
        expected_cost=float(v['expected_cost'].getValue()),
        cvar_cost=cvar_of(cost_s, v['pi'], v['cvar_alpha']),
        cost_s=cost_s,
        grid_cost_s=np.array([v['grid_cost'][s].getValue() for s in range(S)]),
        budget_s=np.array([v['budget'][s].getValue() for s in range(S)]),
        exceedance_s=np.array([sum(v['P_exc'][s][t].X for t in hours) for s in range(S)]),
        import_s=np.array([[v['P_imp'][s][t].X for t in hours] for s in range(S)]),
        member_cost_s=[{i: v['dual_cost'][s][i].getValue() for i in IDs} for s in range(S)],
        ir_slacks={i: v['ir_slack'][i].X for i in v['ir_slack']},
    )