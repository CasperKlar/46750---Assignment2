"""Q3 sensitivity studies. Run from the repository root."""
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from gurobipy import GRB
from src.data_loader import load_community
from src.Q3_models import (make_scenarios, scenario_community,
                           build_stochastic_pricing_problem, cvar_of)
from src.Q2_models import build_pricing_problem

# SETTINGS
ROOT = Path(__file__).resolve().parents[1] / 'results' / 'Q3' / 'base' / 'd1' / 'sensitivity'
N_VALUES = [2, 3, 4, 5, 10]
WEIGHTS = [0, .25, .5, .75, 1]
EPSILONS = [0, .2, .4, .6, .8, 1]
ALPHA = .2
SEED_IN, SEED_OUT = 42, 43
TIME_LIMIT = 300  # seconds per training solve; increase for final experiments
EVAL_LIMIT = 5  # seconds per fixed-price scenario
MIP_GAP = 0.02
TOL = 1e-5


def scenarios():
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / 'scenarios.pkl'
    if not path.exists():
        make_scenarios(ROOT, SEED_IN, SEED_OUT)
    with path.open('rb') as f:
        data = pickle.load(f)
    if data['seeds'] != {'in': SEED_IN, 'out': SEED_OUT}:
        raise ValueError('Saved scenario seeds differ from settings; use a fresh output directory.')
    if len(data['in10']) != 10 or len(data['out100']) < 100:
        raise ValueError('Unexpected scenario counts.')
    return data['in10'], data['out100']


def configure(model, limit, quiet=False):
    model.Params.TimeLimit = limit
    model.Params.MIPGap = MIP_GAP
    model.Params.Seed = 0
    model.Params.OutputFlag = int(not quiet)

def metrics(com, net, price, physical):
    # Reconstruct actual meter flows from net demand, not auxiliary slack variables.
    total = np.sum(net, axis=0)
    imp, exp = np.maximum(total, 0), np.maximum(-total, 0)
    exc = np.maximum(imp - np.asarray(com.contract.capacity_limit_kW), 0)
    beta = com.contract.tariff_discount_ratio
    spot = np.asarray(com.energy_price)
    grid = float((spot + (1-beta)*com.import_tariff) @ imp
                 - (spot - (1-beta)*com.export_tariff) @ exp
                 + com.contract.penalty_DKK_per_kWh * exc.sum())
    return dict(cost_DKK=grid+physical, grid_cost_DKK=grid,
                physical_cost_DKK=physical, cap_exceedance_kWh=float(exc.sum()),
                cap_violation=int(np.any(exc > TOL)),
                hours_above_cap=int(np.sum(exc > TOL)),
                budget_residual_DKK=float(price @ total - grid))


def evaluate(com, sc, price):
    # Optimistic tie-breaking, consistent with Q2/Q3 KKT models.
    # No BB or chance constraint imposed on the realized scenario.
    cs = scenario_community(com, sc)
    m, v = build_pricing_problem(cs, variant='DP0', price_lower=price, price_upper=price)
    try:
        configure(m, EVAL_LIMIT, quiet=True)
        m.optimize()
        row = dict(status=m.Status, optimal=m.Status == GRB.OPTIMAL, runtime_s=m.Runtime)
        if m.SolCount:
            net = [[v['net'][i][t].X for t in v['hours']] for i in v['ids']]
            physical = sum(term.getValue() for term in v['physical_cost_terms'])
            row.update(metrics(cs, net, price, physical))
        return row
    finally:
        m.dispose()


def summarize(rows, prefix):
    df = pd.DataFrame(rows)
    out = {prefix+'count': len(df), prefix+'optimal_count': int(df.optimal.sum())}
    if 'cost_DKK' not in df or df.cost_DKK.isna().any():
        return out  # do not silently average only successful scenarios
    p = np.full(len(df), 1/len(df))
    for col in ['cost_DKK', 'grid_cost_DKK', 'physical_cost_DKK', 'cap_exceedance_kWh',
                'cap_violation', 'budget_residual_DKK']:
        out[prefix+'mean_'+col] = float(df[col].mean())
    out[prefix+'cvar_DKK'] = cvar_of(df.cost_DKK, p, ALPHA)
    out[prefix+'mean_abs_budget_residual_DKK'] = float(df.budget_residual_DKK.abs().mean())
    return out


def run_case(com, pool, out, n, w, eps, folder):
    folder.mkdir(parents=True, exist_ok=True)
    ins = [dict(sc, probability=1/n) for sc in pool[:n]]
    print(f'\n--- N={n}, w={w}, epsilon={eps} ---', flush=True)
    m, v = build_stochastic_pricing_problem(com, ins, requirements='BB',
                  cvar_weight=w, cvar_alpha=ALPHA, chance_epsilon=eps)
    result = dict(N=n, weight=w, epsilon=eps, alpha=ALPHA, seed_in=SEED_IN, seed_out=SEED_OUT)
    try:
        configure(m, TIME_LIMIT)
        m.optimize()
        result.update(status=m.Status, optimal=m.Status == GRB.OPTIMAL,
                      feasible=bool(m.SolCount), runtime_s=m.Runtime)
        if not m.SolCount:
            return result
        result.update(objective_DKK=m.ObjVal, mip_gap=m.MIPGap, objective_bound_DKK=m.ObjBound)
        price = np.array([v['x'][t].X for t in v['hours']])
        pd.DataFrame({'hour': list(v['hours']), 'price_DKK_per_kWh': price}).to_csv(folder/'prices.csv', index=False)
        train = []
        for s, sc in enumerate(ins):
            net = [[v['net'][s][i][t].X for t in v['hours']] for i in v['ids']]
            row = metrics(com, net, price, v['member_cost'][s].getValue())
            row.update(scenario=s, optimal=m.Status == GRB.OPTIMAL,
                       model_cost_DKK=v['cost'][s].getValue(), model_budget_DKK=v['budget'][s].getValue())
            train.append(row)
        pd.DataFrame(train).to_csv(folder/'in_sample.csv', index=False)
        result.update(summarize(train, 'is_'))
        result['is_model_mean_cost_DKK'] = float(np.mean([r['model_cost_DKK'] for r in train]))
        result['is_model_mean_budget_DKK'] = float(np.mean([r['model_budget_DKK'] for r in train]))
    finally:
        m.dispose()
    test = []
    for s, sc in enumerate(out):
        row = evaluate(com, sc, price)
        row['scenario'] = s
        test.append(row)
        pd.DataFrame(test).to_csv(folder/'out_of_sample.csv', index=False)
    result.update(summarize(test, 'oos_'))
    return result


def run_study(name, cases, com, pool, out):
    directory = ROOT / name
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag, n, w, eps in cases:
        folder = directory / tag
        row = run_case(com, pool, out, n, w, eps, folder)
        pd.DataFrame([row]).to_csv(folder/'summary.csv', index=False)
        rows.append(row)
        pd.DataFrame(rows).to_csv(directory/'sensitivity_summary.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', choices=['all','scenario_count','risk_aversion','chance_constraint'], default='all')
    parser.add_argument('--smoke-test', action='store_true', help='One N=2 solve, two test scenarios, separate output folder')
    args = parser.parse_args()
    global ROOT
    if args.smoke_test:
        ROOT = ROOT / 'smoke_test'
    pool, out = scenarios()
    com = load_community('base', day='d1')
    if args.smoke_test:
        run_study('scenario_count', [('N_02', 2, 0, None)], com, pool, out[:2])
        return
    studies = {
        'scenario_count': [(f'N_{n:02d}', n, 0, None) for n in N_VALUES],
        'risk_aversion': [(f'w_{w:.2f}', 10, w, None) for w in WEIGHTS],
        'chance_constraint': [(f'eps_{e:.2f}', 10, 0, e) for e in EPSILONS],
    }
    for name, cases in studies.items():
        if args.study in ('all', name):
            run_study(name, cases, com, pool, out)


if __name__ == '__main__':
    main()
