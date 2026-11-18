import numpy as np
from src import scenario_generation as sg

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

import numpy as np
from src import scenario_generation as sg


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