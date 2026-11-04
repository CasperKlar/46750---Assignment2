"""Question 2: uniform-price Stackelberg model DP, reusable in Question 3.

Requires gurobipy (Gurobi). Uses the existing src.data_loader CommunityData.

The follower LP is replaced by its KKT conditions. Gurobi indicator
constraints enforce complementary slackness exactly, avoiding guessed
big-M constants. The leader chooses the best follower optimum (optimistic DP).

Model variants:
    'DP0'   : price only, no budget-balance or individual-rationality rule
    'DPBB'  : price plus daily budget balance
    'DPBBIR': budget balance and (soft or hard) individual rationality

All times are one-hour slots, so kWh and kW have the same numerical value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import gurobipy as gp
from gurobipy import GRB


@dataclass
class PricingResult:
    status: int
    objective_DKK: float
    grid_cost_DKK: float
    prices_DKK_per_kWh: np.ndarray
    import_kWh: np.ndarray
    export_kWh: np.ndarray
    exceedance_kWh: np.ndarray
    net_by_member_kWh: dict[str, np.ndarray]
    payments_DKK: dict[str, float]
    member_costs_DKK: dict[str, float]
    ir_slacks_DKK: dict[str, float]


def build_pricing_problem(
    community,
    variant: str = 'DP0',
    benchmark_R: Mapping[str, float] | None = None,
    benchmark_I: Mapping[str, float] | None = None,
    ir_soft: bool = True,
    ir_penalty_DKK: float = 1e4,
    ir_tolerance_DKK: float = 1e-5,
    price_lower: np.ndarray | None = None,
    price_upper: np.ndarray | None = None,
    end_soc_rule: str = 'initial',
):
    """Build a Gurobi MILP for the deterministic Q2 pricing problem.

    Parameters
    ----------
    community : CommunityData
        `load_community('base', day='d1')` or a Q3 scenario copy.
    variant : {'DP0', 'DPBB', 'DPBBIR'}
        Adds budget balance and optionally individual rationality.
    benchmark_R / benchmark_I : dict[str, float]
        Each member's *full* cost in benchmark R and I, required for DPBBIR.
    ir_soft : bool
        If True, permit penalized IR violations. Otherwise enforce hard IR.
    end_soc_rule : {'initial', 'none'}
        'initial' requires final battery SoC >= initial SoC. 'none' leaves
        end-of-day SoC free (subject to the capacity bounds).

    Returns
    -------
    (model, variables)
        Build once, call model.optimize(), then extract_pricing_result(...).
    """
    if variant not in {'DP0', 'DPBB', 'DPBBIR'}:
        raise ValueError('variant must be DP0, DPBB or DPBBIR')
    if end_soc_rule not in {'initial', 'none'}:
        raise ValueError('end_soc_rule must be initial or none')
    if variant == 'DPBBIR' and (benchmark_R is None or benchmark_I is None):
        raise ValueError('DPBBIR requires per-member benchmark_R and benchmark_I costs')

    members = community.members
    T = len(community.energy_price)
    hours = range(T)
    IDs = [m.consumer_id for m in members]
    contract = community.contract

    spot = np.asarray(community.energy_price, dtype=float)
    beta = float(contract.tariff_discount_ratio)
    alpha = float(contract.penalty_DKK_per_kWh)
    cap = np.asarray(contract.capacity_limit_kW, dtype=float)
    p_imp = spot + (1 - beta) * community.import_tariff
    p_exp = spot - (1 - beta) * community.export_tariff
    low = p_exp if price_lower is None else np.asarray(price_lower, dtype=float)
    high = p_imp + alpha if price_upper is None else np.asarray(price_upper, dtype=float)
    if len(low) != T or len(high) != T or np.any(low > high):
        raise ValueError('Invalid price bounds')

    model = gp.Model(f'Pricing_{variant}_{community.day}')

    x = model.addVars(hours, lb={t: float(low[t]) for t in hours},
                      ub={t: float(high[t]) for t in hours}, name='x')
    Pimp = model.addVars(hours, lb=0, name='Pimp')
    Pexp = model.addVars(hours, lb=0, name='Pexp')
    Pexc = model.addVars(hours, lb=0, name='Pexc')

    # Each complementary pair (slack >= 0, multiplier >= 0) is enforced by
    # binary z: z=1 => slack=0; z=0 => multiplier=0.
    pair_counter = [0]
    def perp(slack, multiplier, prefix):
        pair_counter[0] += 1
        z = model.addVar(vtype=GRB.BINARY, name=f'{prefix}_z{pair_counter[0]}')
        model.addGenConstrIndicator(z, 1, slack == 0, name=f'{prefix}_slack')
        model.addGenConstrIndicator(z, 0, multiplier == 0, name=f'{prefix}_dual')

    net = {}
    D = {}
    payments = {}
    follower_variables = {}
    physical_cost_terms = []

    for m in members:
        i = m.consumer_id
        ref = np.asarray(m.reference_load, dtype=float)
        lower = np.asarray(m.load_lower_kWh, dtype=float)
        upper = np.asarray(m.load_upper_kWh, dtype=float)
        pv_avail = np.asarray(m.pv_available, dtype=float)
        cL = float(m.linear_disutility)
        cPV = float(m.pv_marginal_cost)
        has_pv = bool(m.has_pv)
        has_battery = bool(m.has_battery)

        l = model.addVars(hours, lb=0, name=f'{i}_l')
        dp = model.addVars(hours, lb=0, name=f'{i}_dplus')
        dm = model.addVars(hours, lb=0, name=f'{i}_dminus')
        q = model.addVars(hours, lb=-GRB.INFINITY, name=f'{i}_q')
        g = model.addVars(hours, lb=0, name=f'{i}_pv') if has_pv else None
        net[i] = q

        # Dual variables of the two deviation constraints and load bounds.
        xp = model.addVars(hours, lb=0, name=f'{i}_xiPlus')
        xm = model.addVars(hours, lb=0, name=f'{i}_xiMinus')
        lm = model.addVars(hours, lb=0, name=f'{i}_muLoadMin')
        lM = model.addVars(hours, lb=0, name=f'{i}_muLoadMax')
        eMin = model.addVar(lb=0, name=f'{i}_muEnergyMin')
        pv_m = model.addVars(hours, lb=0, name=f'{i}_muPvMin') if has_pv else None
        pv_M = model.addVars(hours, lb=0, name=f'{i}_muPvMax') if has_pv else None

        if has_battery:
            charge = model.addVars(hours, lb=0, name=f'{i}_charge')
            discharge = model.addVars(hours, lb=0, name=f'{i}_discharge')
            soc = model.addVars(hours, lb=0, name=f'{i}_soc')
            rho = model.addVars(hours, lb=-GRB.INFINITY, name=f'{i}_rho')
            sm = model.addVars(hours, lb=0, name=f'{i}_muSocMin')
            sM = model.addVars(hours, lb=0, name=f'{i}_muSocMax')
            cm = model.addVars(hours, lb=0, name=f'{i}_muChargeMin')
            cM = model.addVars(hours, lb=0, name=f'{i}_muChargeMax')
            bm = model.addVars(hours, lb=0, name=f'{i}_muDischargeMin')
            bM = model.addVars(hours, lb=0, name=f'{i}_muDischargeMax')
            S = float(m.battery_capacity_kWh)
            Pc = float(m.battery_max_charge_kW)
            Pd = float(m.battery_max_discharge_kW)
            ec = float(m.battery_charging_efficiency)
            ed = float(m.battery_discharging_efficiency)
            initial = float(m.battery_initial_soc_kWh)
        else:
            charge = discharge = soc = rho = None
            sm = sM = cm = cM = bm = bM = None
            S = Pc = Pd = initial = 0.0

        for t in hours:
            # Primal feasibility and complementarity.
            a = dp[t] - l[t] + float(ref[t])
            b = dm[t] + l[t] - float(ref[t])
            model.addConstr(a >= 0, name=f'{i}_devPlus[{t}]')
            model.addConstr(b >= 0, name=f'{i}_devMinus[{t}]')
            perp(a, xp[t], f'{i}_devPlus[{t}]')  # Only one can be positive using perp() below.
            perp(b, xm[t], f'{i}_devMinus[{t}]')
            model.addConstr(l[t] >= float(lower[t]))
            model.addConstr(l[t] <= float(upper[t]))
            perp(l[t] - float(lower[t]), lm[t], f'{i}_lmin[{t}]')
            perp(float(upper[t]) - l[t], lM[t], f'{i}_lmax[{t}]')
            perp(dp[t], cL - xp[t], f'{i}_dplus[{t}]')
            perp(dm[t], cL - xm[t], f'{i}_dminus[{t}]')
            model.addConstr(xp[t] <= cL)
            model.addConstr(xm[t] <= cL)

            # Load stationarity, with lambda_{i,t} = x_t.
            model.addConstr(-xp[t] + xm[t] + lm[t] - lM[t] + eMin - x[t] == 0,  #xp, xm (deviation constraints), lm, lM (upper and lower load limits), eMin (min daily energy), x[t] member energy balance (see Table 4 in LateX)
                            name=f'{i}_station_l[{t}]')
            pv_term = g[t] if has_pv else 0
            ch_term = charge[t] if has_battery else 0
            dis_term = discharge[t] if has_battery else 0
            model.addConstr(q[t] - l[t] + pv_term - ch_term + dis_term == 0,
                            name=f'{i}_balance[{t}]')

            if has_pv:
                model.addConstr(g[t] <= float(pv_avail[t]))
                perp(g[t], pv_m[t], f'{i}_pvmin[{t}]')
                perp(float(pv_avail[t]) - g[t], pv_M[t], f'{i}_pvmax[{t}]')
                model.addConstr(cPV == pv_m[t] - pv_M[t] + x[t],
                                name=f'{i}_station_pv[{t}]')

            if has_battery:
                prev_soc = initial if t == 0 else soc[t - 1]
                model.addConstr(soc[t] - prev_soc - ec * charge[t]
                                + discharge[t] / ed == 0, name=f'{i}_soc_balance[{t}]')
                min_soc = initial if (end_soc_rule == 'initial' and t == T - 1) else 0.0
                model.addConstr(soc[t] >= min_soc)
                model.addConstr(soc[t] <= S)
                model.addConstr(charge[t] <= Pc)
                model.addConstr(discharge[t] <= Pd)
                perp(soc[t] - min_soc, sm[t], f'{i}_socmin[{t}]')
                perp(S - soc[t], sM[t], f'{i}_socmax[{t}]')
                perp(charge[t], cm[t], f'{i}_chmin[{t}]')
                perp(Pc - charge[t], cM[t], f'{i}_chmax[{t}]')
                perp(discharge[t], bm[t], f'{i}_dismin[{t}]')
                perp(Pd - discharge[t], bM[t], f'{i}_dismax[{t}]')
                model.addConstr(cm[t] - cM[t] - x[t] - ec * rho[t] == 0,
                                name=f'{i}_station_charge[{t}]')
                model.addConstr(bm[t] - bM[t] + x[t] + rho[t] / ed == 0,
                                name=f'{i}_station_discharge[{t}]')
                next_rho = rho[t + 1] if t + 1 < T else 0
                model.addConstr(rho[t] - next_rho + sm[t] - sM[t] == 0,
                                name=f'{i}_station_soc[{t}]')

        Emin = float(m.min_daily_energy_kWh)
        total_load = gp.quicksum(l[t] for t in hours)
        model.addConstr(total_load >= Emin, name=f'{i}_daily_energy')
        perp(total_load - Emin, eMin, f'{i}_dailyEnergy')

        # Strong-duality RHS D_i; avoids bilinear posted-price x net-energy.
        dual = gp.LinExpr(Emin * eMin)
        for t in hours:
            dual += (-float(ref[t]) * xp[t] + float(ref[t]) * xm[t]
                     + float(lower[t]) * lm[t] - float(upper[t]) * lM[t])
            if has_pv:
                dual += -float(pv_avail[t]) * pv_M[t]
            if has_battery:
                min_soc = initial if (end_soc_rule == 'initial' and t == T - 1) else 0.0
                dual += min_soc * sm[t] - S * sM[t] - Pc * cM[t] - Pd * bM[t]
        if has_battery:
            dual += initial * rho[0]
        D[i] = dual

        internal_cost = gp.quicksum(cL * (dp[t] + dm[t])
                                    + (cPV * g[t] if has_pv else 0)
                                    for t in hours)
        physical_cost_terms.append(internal_cost)
        payments[i] = dual - internal_cost
        follower_variables[i] = dict(load=l, dplus=dp, dminus=dm, pv=g,
                                     charge=charge, discharge=discharge, soc=soc, net=q)

    # Upper-level community meter and DSO contract.
    for t in hours:
        model.addConstr(gp.quicksum(net[i][t] for i in IDs) == Pimp[t] - Pexp[t],
                        name=f'community_balance[{t}]')
        model.addConstr(Pexc[t] >= Pimp[t] - float(cap[t]),
                        name=f'cap_exceedance[{t}]')
        if contract.export_capacity_limit_kW is not None:
            model.addConstr(Pexp[t] <= float(contract.export_capacity_limit_kW))

    grid_cost = gp.quicksum(float(p_imp[t]) * Pimp[t]
                            - float(p_exp[t]) * Pexp[t]
                            + alpha * Pexc[t] for t in hours)
    ir_slack = {}

    if variant in {'DPBB', 'DPBBIR'}: # Extra scenarios in Question 2
        model.addConstr(gp.quicksum(payments[i] for i in IDs) == grid_cost,
                        name='daily_budget_balance')
    if variant == 'DPBBIR':
        for i in IDs:
            limit = min(float(benchmark_R[i]), float(benchmark_I[i]))
            if ir_soft:
                ir_slack[i] = model.addVar(lb=0, name=f'{i}_ir_slack')
                model.addConstr(D[i] <= limit + ir_tolerance_DKK + ir_slack[i],
                                name=f'{i}_IR')
            else:
                model.addConstr(D[i] <= limit + ir_tolerance_DKK, name=f'{i}_IR')

    objective = grid_cost + gp.quicksum(physical_cost_terms)
    if ir_soft and variant == 'DPBBIR':
        objective += ir_penalty_DKK * gp.quicksum(ir_slack.values())
    model.setObjective(objective, GRB.MINIMIZE)

    variables = dict(x=x, Pimp=Pimp, Pexp=Pexp, Pexc=Pexc, net=net,
                     dual_cost=D, payment=payments, grid_cost=grid_cost,
                     ir_slack=ir_slack, follower=follower_variables, ids=IDs,
                     hours=hours)
    return model, variables


def extract_pricing_result(model, v):
    """Extract numeric results after optimize() has found a feasible solution."""
    if model.SolCount == 0:
        raise RuntimeError(f'No solution found. Gurobi status: {model.Status}')
    return PricingResult(
        status=model.Status,
        objective_DKK=model.ObjVal,
        grid_cost_DKK=v['grid_cost'].getValue(),
        prices_DKK_per_kWh=np.array([v['x'][t].X for t in v['hours']]),
        import_kWh=np.array([v['Pimp'][t].X for t in v['hours']]),
        export_kWh=np.array([v['Pexp'][t].X for t in v['hours']]),
        exceedance_kWh=np.array([v['Pexc'][t].X for t in v['hours']]),
        net_by_member_kWh={i: np.array([v['net'][i][t].X for t in v['hours']])
                           for i in v['ids']},
        payments_DKK={i: v['payment'][i].getValue() for i in v['ids']},
        member_costs_DKK={i: v['dual_cost'][i].getValue() for i in v['ids']},
        ir_slacks_DKK={i: v['ir_slack'][i].X for i in v['ir_slack']},
    )


if __name__ == '__main__':
    from src.data_loader import load_community

    community = load_community('base', day='d1')
    model, vars_ = build_pricing_problem(community, variant='DP0')
    model.Params.TimeLimit = 300  # optional, seconds
    model.Params.MIPGap = 0.01
    model.optimize()
    if model.SolCount:
        result = extract_pricing_result(model, vars_)
        print('Community cost [DKK]:', round(result.objective_DKK, 2))
        print('Internal prices [DKK/kWh]:', np.round(result.prices_DKK_per_kWh, 3))
        print('Grid cost [DKK]:', round(result.grid_cost_DKK, 2))
