import pickle, os                     
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