import pickle  
from dataclasses import replace
import numpy as np
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from src.data_loader import load_community         
from src import scenario_generation as sg  
from src.config import END_OF_DAY_CHARGE      


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

def build_stochastic_pricing_problem(
    community,
    scenarios,
    requirements: str = 'BB',
    cvar_weight: float = 0.0,
    cvar_alpha: float = 0.2,
    benchmark_R=None,
    benchmark_I=None,
    ir_soft: bool = False,
    ir_penalty_DKK: float = 1e4,
    ir_tolerance_DKK: float = 1e-5,
    price_lower=None,
    price_upper=None,
    end_E_bat_rule: str = 'final',
):
    """Build the stochastic pricing MILP SDP (one price vector, one KKT system per member and scenario).

    The manager posts ONE price vector x before theta is known. In every scenario s each member
    responds optimally to x (KKT conditions of its LP, using that scenario's reference load,
    load bounds and daily minimum). The structure of each member block is identical to
    Q2_models.build_pricing_problem; only the names carry a scenario index.

    Parameters
    ----------
    community : CommunityData
        Base community (theta = 1), from load_community.
    scenarios : list[dict]
        Joint scenarios (in5, in10, ...). Uses sc["probability"] and sc["members"].
    requirements : {'none', 'BB', 'BBIR'}
        'none' : no requirements        (stochastic DP0)
        'BB'   : budget balance in expectation:  sum_s pi_s * B_s = 0
        'BBIR' : BB plus IR in expectation:  sum_s pi_s * D_{i,s} <= min(J_R, J_I) (+ slack if ir_soft)
    cvar_weight : float in [0, 1]
        Weight w of the CVaR in the objective. 0 = risk-neutral base model SDP^RN.
    cvar_alpha : float in (0, 1]
        Tail share of the CVaR (0.2 with 5 scenarios = the single worst scenario).
    benchmark_R / benchmark_I : dict[str, float]
        Members' costs in R and I (Q1), required for 'BBIR'. Fixed at theta = 1.
    ir_soft : bool
        If True, IR may be violated at a penalty ir_penalty_DKK per DKK of slack.
    price_lower / price_upper : array-like or None
        Price bounds (default: discounted export price to discounted import price + penalty, as in Q2).

    Objective (minimized):
        (1 - w) * E[C_s]  +  w * ( zeta + 1/alpha * sum_s pi_s * eta_s ),  eta_s >= C_s - zeta, eta_s >= 0
    with C_s = grid cost + members' physical cost in scenario s (same definition as Q2's objective).

    Returns
    -------
    (model, variables)
    """
    if requirements not in {'none', 'BB', 'BBIR'}:
        raise ValueError("requirements must be 'none', 'BB' or 'BBIR'")
    if requirements == 'BBIR' and (benchmark_R is None or benchmark_I is None):
        raise ValueError('BBIR requires benchmark_R and benchmark_I')
    if not 0.0 <= cvar_weight <= 1.0:
        raise ValueError('cvar_weight must be in [0, 1]')
    if not 0.0 < cvar_alpha <= 1.0:
        raise ValueError('cvar_alpha must be in (0, 1]')
    if end_E_bat_rule not in {'final', 'initial', 'none'}:
        raise ValueError("end_E_bat_rule must be 'final', 'initial' or 'none'")

    S = len(scenarios)
    pi = [float(sc.get('probability', 1.0 / S)) for sc in scenarios]
    comms = [scenario_community(community, sc) for sc in scenarios]   # one community per scenario

    c0 = community
    T = len(c0.energy_price)
    hours = range(T)
    IDs = [m.consumer_id for m in c0.members]
    contract = c0.contract

    # contract and price bounds: identical to Q2 (prices, tariffs and cap are not uncertain)
    spot = np.asarray(c0.energy_price, dtype=float)
    beta = float(contract.tariff_discount_ratio)
    alpha = float(contract.penalty_DKK_per_kWh)
    cap = np.asarray(contract.capacity_limit_kW, dtype=float)
    p_imp = spot + (1 - beta) * c0.import_tariff
    p_exp = spot - (1 - beta) * c0.export_tariff
    low = p_exp if price_lower is None else np.asarray(price_lower, dtype=float)
    high = p_imp + alpha if price_upper is None else np.asarray(price_upper, dtype=float)
    if len(low) != T or len(high) != T or np.any(low > high):
        raise ValueError('Invalid price bounds')

    model = gp.Model(f'SDP_{requirements}_S{S}_{c0.day}')

    # ---- first-stage decision: ONE price vector shared by all scenarios ----
    x = model.addVars(hours, lb={t: float(low[t]) for t in hours},
                      ub={t: float(high[t]) for t in hours}, name='x')

    # complementarity pair (slack >= 0, multiplier >= 0) with a binary, as in Q2
    pair_counter = [0]
    def perp(slack, multiplier, prefix):
        pair_counter[0] += 1
        z = model.addVar(vtype=GRB.BINARY, name=f'{prefix}_z{pair_counter[0]}')
        model.addGenConstrIndicator(z, 1, slack == 0, name=f'{prefix}_slack')
        model.addGenConstrIndicator(z, 0, multiplier == 0, name=f'{prefix}_dual')

    # scenario-indexed containers
    P_imp, P_exp, P_exc = [], [], []
    net_s, D_s, Pi_s = [], [], []
    grid_cost_s, member_cost_s, cost_s, budget_s = [], [], [], []
    follower_s = []

    for s in range(S):
        com_s = comms[s]

        # second-stage (scenario) grid variables
        P_imp.append(model.addVars(hours, lb=0, name=f's{s}_P_imp'))
        P_exp.append(model.addVars(hours, lb=0, name=f's{s}_P_exp'))
        P_exc.append(model.addVars(hours, lb=0, name=f's{s}_P_exc'))

        net, D, Pi, follower_vars, phys_terms = {}, {}, {}, {}, []

        for m in com_s.members:
            i = m.consumer_id
            pre = f's{s}_{i}'                                   # name prefix with scenario index

            # scenario-specific member data (theta enters here)
            l_ref = np.asarray(m.reference_load, dtype=float)
            l_lower = np.asarray(m.load_lower_kWh, dtype=float)
            l_upper = np.asarray(m.load_upper_kWh, dtype=float)
            pv_avail = np.asarray(m.pv_available, dtype=float)
            c_L = float(m.linear_disutility)
            c_PV = float(m.pv_marginal_cost)
            has_pv = bool(m.has_pv)
            has_battery = bool(m.has_battery)
            E_min = float(m.min_daily_energy_kWh)

            # primal variables
            l = model.addVars(hours, lb=0, name=f'{pre}_l')
            d_pos = model.addVars(hours, lb=0, name=f'{pre}_dplus')
            d_neg = model.addVars(hours, lb=0, name=f'{pre}_dminus')
            q = model.addVars(hours, lb=-GRB.INFINITY, name=f'{pre}_q')
            g = model.addVars(hours, lb=0, name=f'{pre}_pv') if has_pv else None
            net[i] = q

            # dual variables
            xi_pos = model.addVars(hours, lb=0, name=f'{pre}_xiPlus')
            xi_neg = model.addVars(hours, lb=0, name=f'{pre}_xiMinus')
            mu_L_min = model.addVars(hours, lb=0, name=f'{pre}_muLoadMin')
            mu_L_max = model.addVars(hours, lb=0, name=f'{pre}_muLoadMax')
            mu_E_min = model.addVar(lb=0, name=f'{pre}_muEnergyMin')
            mu_PV_min = model.addVars(hours, lb=0, name=f'{pre}_muPvMin') if has_pv else None
            mu_PV_max = model.addVars(hours, lb=0, name=f'{pre}_muPvMax') if has_pv else None

            if has_battery:
                s_ch = model.addVars(hours, lb=0, name=f'{pre}_s_ch')
                s_dis = model.addVars(hours, lb=0, name=f'{pre}_s_dis')
                E_bat = model.addVars(hours, lb=0, name=f'{pre}_E_bat')
                rho = model.addVars(hours, lb=-GRB.INFINITY, name=f'{pre}_rho')
                mu_bat_min = model.addVars(hours, lb=0, name=f'{pre}_muSocMin')
                mu_bat_max = model.addVars(hours, lb=0, name=f'{pre}_muSocMax')
                mu_ch_min = model.addVars(hours, lb=0, name=f'{pre}_muChargeMin')
                mu_ch_max = model.addVars(hours, lb=0, name=f'{pre}_muChargeMax')
                mu_dis_min = model.addVars(hours, lb=0, name=f'{pre}_muDischargeMin')
                mu_dis_max = model.addVars(hours, lb=0, name=f'{pre}_muDischargeMax')
                Sb = float(m.battery_capacity_kWh)
                P_ch = float(m.battery_max_charge_kW)
                P_dis = float(m.battery_max_discharge_kW)
                eta_ch = float(m.battery_charging_efficiency)
                eta_dis = float(m.battery_discharging_efficiency)
                initial = float(m.battery_initial_soc_kWh)
                final_soc = float(m.battery_final_soc_kWh if m.battery_final_soc_kWh is not None
                                  else END_OF_DAY_CHARGE)
            else:
                s_ch = s_dis = E_bat = rho = None
                mu_bat_min = mu_bat_max = mu_ch_min = mu_ch_max = mu_dis_min = mu_dis_max = None
                Sb = P_ch = P_dis = initial = final_soc = 0.0

            def soc_lower_bound(t):
                if t != T - 1:
                    return 0.0
                if end_E_bat_rule == 'final':
                    return final_soc
                if end_E_bat_rule == 'initial':
                    return initial
                return 0.0

            for t in hours:
                # deviation constraints + complementarity
                a = d_pos[t] - l[t] + float(l_ref[t])
                b = d_neg[t] + l[t] - float(l_ref[t])
                model.addConstr(a >= 0, name=f'{pre}_devPlus[{t}]')
                model.addConstr(b >= 0, name=f'{pre}_devMinus[{t}]')
                perp(a, xi_pos[t], f'{pre}_devPlus[{t}]')
                perp(b, xi_neg[t], f'{pre}_devMinus[{t}]')

                # load bounds (scenario-specific)
                model.addConstr(l[t] >= float(l_lower[t]))
                model.addConstr(l[t] <= float(l_upper[t]))
                perp(l[t] - float(l_lower[t]), mu_L_min[t], f'{pre}_lmin[{t}]')
                perp(float(l_upper[t]) - l[t], mu_L_max[t], f'{pre}_lmax[{t}]')
                perp(d_pos[t], c_L - xi_pos[t], f'{pre}_dplus[{t}]')
                perp(d_neg[t], c_L - xi_neg[t], f'{pre}_dminus[{t}]')
                model.addConstr(xi_pos[t] <= c_L)
                model.addConstr(xi_neg[t] <= c_L)

                # load stationarity with lambda_{i,t,s} = x_t  (this ties all scenarios to ONE price)
                model.addConstr(xi_pos[t] - xi_neg[t] - mu_L_min[t] + mu_L_max[t] - mu_E_min + x[t] == 0,
                                name=f'{pre}_station_l[{t}]')

                # member energy balance
                pv_term = g[t] if has_pv else 0
                ch_term = s_ch[t] if has_battery else 0
                dis_term = s_dis[t] if has_battery else 0
                model.addConstr(q[t] - l[t] + pv_term - ch_term + dis_term == 0,
                                name=f'{pre}_balance[{t}]')

                if has_pv:
                    model.addConstr(g[t] <= float(pv_avail[t]))
                    perp(g[t], mu_PV_min[t], f'{pre}_pvmin[{t}]')
                    perp(float(pv_avail[t]) - g[t], mu_PV_max[t], f'{pre}_pvmax[{t}]')
                    model.addConstr(c_PV == mu_PV_min[t] - mu_PV_max[t] + x[t],
                                    name=f'{pre}_station_pv[{t}]')

                if has_battery:
                    prev_E_bat = initial if t == 0 else E_bat[t - 1]
                    model.addConstr(E_bat[t] - prev_E_bat - eta_ch * s_ch[t]
                                    + s_dis[t] / eta_dis == 0, name=f'{pre}_soc_balance[{t}]')
                    min_E_bat = soc_lower_bound(t)
                    model.addConstr(E_bat[t] >= min_E_bat)
                    model.addConstr(E_bat[t] <= Sb)
                    model.addConstr(s_ch[t] <= P_ch)
                    model.addConstr(s_dis[t] <= P_dis)
                    perp(E_bat[t] - min_E_bat, mu_bat_min[t], f'{pre}_socmin[{t}]')
                    perp(Sb - E_bat[t], mu_bat_max[t], f'{pre}_socmax[{t}]')
                    perp(s_ch[t], mu_ch_min[t], f'{pre}_chmin[{t}]')
                    perp(P_ch - s_ch[t], mu_ch_max[t], f'{pre}_chmax[{t}]')
                    perp(s_dis[t], mu_dis_min[t], f'{pre}_dismin[{t}]')
                    perp(P_dis - s_dis[t], mu_dis_max[t], f'{pre}_dismax[{t}]')
                    model.addConstr(-mu_ch_min[t] + mu_ch_max[t] + x[t] - eta_ch * rho[t] == 0,
                                    name=f'{pre}_station_charge[{t}]')
                    model.addConstr(-mu_dis_min[t] + mu_dis_max[t] - x[t] + rho[t] / eta_dis == 0,
                                    name=f'{pre}_station_discharge[{t}]')
                    next_rho = rho[t + 1] if t + 1 < T else 0
                    model.addConstr(rho[t] - next_rho - mu_bat_min[t] + mu_bat_max[t] == 0,
                                    name=f'{pre}_station_soc[{t}]')

            # daily minimum energy (scenario-specific E_min)
            total_load = gp.quicksum(l[t] for t in hours)
            model.addConstr(total_load >= E_min, name=f'{pre}_daily_energy')
            perp(total_load - E_min, mu_E_min, f'{pre}_dailyEnergy')

            # strong-duality objective D_{i,s}: avoids the bilinear x_t * q_{i,t,s}
            dual = gp.LinExpr(E_min * mu_E_min)
            for t in hours:
                dual += (-float(l_ref[t]) * xi_pos[t] + float(l_ref[t]) * xi_neg[t]
                         + float(l_lower[t]) * mu_L_min[t] - float(l_upper[t]) * mu_L_max[t])
                if has_pv:
                    dual += -float(pv_avail[t]) * mu_PV_max[t]
                if has_battery:
                    dual += (soc_lower_bound(t) * mu_bat_min[t] - Sb * mu_bat_max[t]
                             - P_ch * mu_ch_max[t] - P_dis * mu_dis_max[t])
            if has_battery:
                dual -= initial * rho[0]
            D[i] = dual

            internal_cost = gp.quicksum(c_L * (d_pos[t] + d_neg[t]) + (c_PV * g[t] if has_pv else 0)
                                        for t in hours)
            phys_terms.append(internal_cost)
            Pi[i] = dual - internal_cost                         # linearized payment Pi_{i,s}
            follower_vars[i] = dict(load=l, dplus=d_pos, dminus=d_neg, pv=g,
                                    s_ch=s_ch, s_dis=s_dis, E_bat=E_bat, net=q)

        # community meter and DSO contract in scenario s
        for t in hours:
            model.addConstr(gp.quicksum(net[i][t] for i in IDs) == P_imp[s][t] - P_exp[s][t],
                            name=f's{s}_community_balance[{t}]')
            model.addConstr(P_exc[s][t] >= P_imp[s][t] - float(cap[t]),
                            name=f's{s}_cap_exceedance[{t}]')
            if contract.export_capacity_limit_kW is not None:
                model.addConstr(P_exp[s][t] <= float(contract.export_capacity_limit_kW))

        g_cost = gp.quicksum(float(p_imp[t]) * P_imp[s][t] - float(p_exp[t]) * P_exp[s][t]
                             + alpha * P_exc[s][t] for t in hours)
        m_cost = gp.quicksum(phys_terms)

        net_s.append(net); D_s.append(D); Pi_s.append(Pi); follower_s.append(follower_vars)
        grid_cost_s.append(g_cost)
        member_cost_s.append(m_cost)
        cost_s.append(g_cost + m_cost)                           # C_s, same as Q2's community cost
        budget_s.append(gp.quicksum(Pi[i] for i in IDs) - g_cost)   # B_s = payments - grid cost

    # ---- requirements in expectation ----
    ir_slack = {}
    if requirements in {'BB', 'BBIR'}:
        model.addConstr(gp.quicksum(pi[s] * budget_s[s] for s in range(S)) == 0,
                        name='budget_balance_in_expectation')
    if requirements == 'BBIR':
        for i in IDs:
            limit = min(float(benchmark_R[i]), float(benchmark_I[i]))
            lhs = gp.quicksum(pi[s] * D_s[s][i] for s in range(S))   # expected cost of member i
            if ir_soft:
                ir_slack[i] = model.addVar(lb=0, name=f'{i}_ir_slack')
                model.addConstr(lhs <= limit + ir_tolerance_DKK + ir_slack[i], name=f'{i}_IR_exp')
            else:
                model.addConstr(lhs <= limit + ir_tolerance_DKK, name=f'{i}_IR_exp')

    # ---- objective: (1-w) E[C] + w CVaR_alpha[C] (Rockafellar-Uryasev) ----
    expected_cost = gp.quicksum(pi[s] * cost_s[s] for s in range(S))
    zeta, eta = None, None
    if cvar_weight > 0:
        zeta = model.addVar(lb=-GRB.INFINITY, name='cvar_zeta')       # settles at the VaR of the cost
        eta = model.addVars(S, lb=0, name='cvar_eta')                 # cost excess of scenario s above zeta
        for s in range(S):
            model.addConstr(eta[s] >= cost_s[s] - zeta, name=f'cvar_eta[{s}]')
        cvar = zeta + gp.quicksum(pi[s] * eta[s] for s in range(S)) / cvar_alpha
        objective = (1 - cvar_weight) * expected_cost + cvar_weight * cvar
    else:
        objective = expected_cost
    if ir_soft and requirements == 'BBIR':
        objective += ir_penalty_DKK * gp.quicksum(ir_slack.values())
    model.setObjective(objective, GRB.MINIMIZE)

    variables = dict(x=x, S=S, pi=pi, ids=IDs, hours=hours,
                     P_imp=P_imp, P_exp=P_exp, P_exc=P_exc, net=net_s,
                     dual_cost=D_s, payment=Pi_s, grid_cost=grid_cost_s,
                     member_cost=member_cost_s, cost=cost_s, budget=budget_s,
                     expected_cost=expected_cost, zeta=zeta, eta=eta,
                     ir_slack=ir_slack, follower=follower_s,
                     cvar_weight=cvar_weight, cvar_alpha=cvar_alpha)
    return model, variables


def cvar_of(costs, probs, alpha):
    """CVaR_alpha of a cost distribution: mean cost of the worst alpha share of the probability mass.

    Args:
        costs: array-like, cost per scenario [DKK].
        probs: array-like, probability per scenario.
        alpha: float in (0, 1], tail share.
    Returns:
        float: CVaR [DKK]. alpha = 1 gives the mean.
    """
    costs = np.asarray(costs, dtype=float)
    probs = np.asarray(probs, dtype=float)
    order = np.argsort(-costs)                       # worst (highest cost) first
    remaining, total = alpha, 0.0
    for k in order:
        take = min(probs[k], remaining)
        total += take * costs[k]
        remaining -= take
        if remaining <= 1e-12:
            break
    return total / alpha

