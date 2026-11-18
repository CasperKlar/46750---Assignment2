import pickle  
from dataclasses import replace
import numpy as np
import numpy as np

from src.data_loader import load_community         
from src import scenario_generation as sg        


def make_scenarios(save_dir,SEED_IN, SEED_OUT ):
    """Draw the in-sample and out-of-sample scenario sets of Question 3 and save them.

    Args:
        save_dir: Path or str, folder where scenarios.pkl is written.
        SEED_IN: int, seed of the in-sample sets (N=5 and N=10).
        SEED_OUT: int, seed of the out-of-sample set (100 scenarios).

    Returns:
        tuple: (ins5, ins10, out100), three lists of joint scenarios. Each scenario is a
        dict with keys "id", "probability", "index", "theta" and "members".
        The same three sets and the seeds are saved to save_dir/scenarios.pkl.
    """

    com = load_community("base") # load the 16-member base community on day d1     

    # app = appliance catalogue (loads, PV, battery); par = which member has what
    # The scenario code needs both as raw dictionaries.              
    app, par = com.raw["appliance_params"], com.raw["params"]

    # the uncertainty rules: bounds [0.8, 1.2], mean 1, kappa 2                                              
    rules = sg.load_json("preference_uncertainty_params.json")
                                                   
    # 5 in-sample scenarios: the base set the prices are optimized on
    ins5   = sg.draw_preference_scenarios(app, par, rules, n=5,   seed=SEED_IN)
    # 10 in-sample scenarios, same seed. Needed for the scenario-count study                                               
    ins10  = sg.draw_preference_scenarios(app, par, rules, n=10,  seed=SEED_IN)
    # 100 out-of-sample scenarios, different seed: used only to test the prices,                                               
    out100 = sg.draw_preference_scenarios(app, par, rules, n=100, seed=SEED_OUT)

    # save results
    with open(f"{save_dir}/scenarios.pkl", "wb") as f: 
        pickle.dump({"in5": ins5, "in10": ins10, "out100": out100,
                     "seeds": {"in": SEED_IN, "out": SEED_OUT}}, f)
                                                   
    return ins5, ins10, out100      

def scenario_community(com, sc):
    """Build the community as it looks in one uncertain scenario.

    Four fields change per member: the reference load, the hourly lower and upper
    load bounds (they follow the reference), and the minimum daily energy (it
    equals the scenario's daily reference energy). Prices, tariffs, PV, battery,
    disutility slopes and the DSO contract stay as in the base community. 

    Args:
        com: CommunityData, the base community (theta = 1), from load_community.
        sc: dict, one joint scenario from sg.draw_preference_scenarios. Uses
            sc["members"][consumer_id][appliance_id], the perturbed appliance
            dicts of that scenario.

    Returns:
        CommunityData: a new community with the scenario's reference_load [kWh/h],
        load_lower_kWh and load_upper_kWh [kWh/h] and min_daily_energy_kWh [kWh/day]
        for every member.
    """
    new_members = []
    for m in com.members:
        # perturbed appliance dicts of this member in this scenario
        apps = sc["members"][m.consumer_id]

        # the member's flexible load; its ID starts with "FL_" (PV and battery do not)
        load = next(a for k, a in apps.items() if k.startswith("FL_"))

        # reference profile [kWh/h] = connection capacity x (already theta-scaled) capacity factor
        ref = m.load_max_kWh * np.asarray(load["reference_load_capacity_factor"], dtype=float)

        # maximum allowed hourly deviation from the reference [kWh]
        D = m.max_hourly_deviation_kWh

        new_members.append(replace(
            m,                                                     # copy of the member, only the fields below change
            reference_load=ref,                                    # new preferred profile
            load_upper_kWh=np.minimum(m.load_max_kWh, ref + D),    # upper bound follows the new reference
            load_lower_kWh=np.maximum(m.load_min_kWh, ref - D),    # lower bound follows the new reference
            min_daily_energy_kWh=float(load["min_total_energy_per_day_kWh"]),  # scenario's daily reference energy
        ))

    # same community (prices, contract, ...) but with the scenario's members
    return replace(com, members=new_members)