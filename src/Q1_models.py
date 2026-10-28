from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from dataclasses import dataclass

from .data_loader import CommunityData
from src.config import END_OF_DAY_CHARGE
from .model_validation import validate_solution


@dataclass
class Assignment2Results:
    model_name: str
    status: str
    objective: float
    hourly: pd.DataFrame
    member_results: pd.DataFrame

    # Validation results
    is_optimal: bool
    is_feasible: bool
    max_constraint_violation: float
    max_bound_violation: float
    objective_matches: bool
    validation_passed: bool
    validation_messages: list[str]

class ModelR:
    """Assignment 2, Model R: community optimization."""

    def __init__(self, community: CommunityData, verbose: bool = False):
        self.community = community
        self.verbose = verbose

        self.I = range(len(community.members))
        self.T = range(len(community.energy_price))

        self.m = gp.Model("model_r")
        self.m.Params.OutputFlag = int(verbose)

        self.var = {}
        self.con = {}

    def build(self) -> "ModelR":
        """Create variables, objective, and constraints."""
        # TODO: Add Model I variables.
        m, T, I, c = self.m, self.T, self.I, self.community
        
        # Consumer and Hourly variables
        self.var["l"] = m.addVars(I,T, vtype=GRB.CONTINUOUS, name="l")
        self.var["g"] = m.addVars(I,T, lb=0, vtype=GRB.CONTINUOUS, name="g")
        self.var["q_imp"] = m.addVars(I,T, lb=0, vtype=GRB.CONTINUOUS, name="q_imp")
        self.var["q_exp"] = m.addVars(I,T, lb=0, vtype=GRB.CONTINUOUS, name="q_exp")
        self.var["d_+"] = m.addVars(I,T, lb=0, vtype=GRB.CONTINUOUS, name="d_+")
        self.var["d_-"] = m.addVars(I,T, lb=0, vtype=GRB.CONTINUOUS, name="d_-")
        self.var["s_ch"] = m.addVars(I,T, lb=0, vtype=GRB.CONTINUOUS, name="s_ch")
        self.var["s_dis"] = m.addVars(I,T, lb=0,vtype=GRB.CONTINUOUS, name="s_dis")

        # Consumer variables
        self.var["E_min"] = m.addVars(I, vtype=GRB.CONTINUOUS, name="E_min")
        self.var["PV_max"] = m.addVars(I, vtype=GRB.CONTINUOUS, name="PV_max")
        self.var["S_max"] = m.addVars(I, vtype=GRB.CONTINUOUS, name="S_max")
        self.var["E_end"] = m.addVars(I, vtype=GRB.CONTINUOUS, name="E_end")
        self.var["P_ch_max"] = m.addVars(I, vtype=GRB.CONTINUOUS, name="P_ch_max")
        self.var["P_dis_max"] = m.addVars(I, vtype=GRB.CONTINUOUS, name="P_dis_max")

        # State of charge: 25 states for 24 hourly actions
        #battery needs 25 states to cover the one at the start and end of day
        n_hours = len(T)
        self.var["E_bat"] = m.addVars(I, range(n_hours + 1), lb=0, vtype=GRB.CONTINUOUS, name="E_bat")

        l = self.var["l"]
        g = self.var["g"]
        q_imp = self.var["q_imp"]
        q_exp = self.var["q_exp"]
        d_pos = self.var["d_+"]
        d_neg = self.var["d_-"]
        s_ch = self.var["s_ch"]
        s_dis = self.var["s_dis"]
        E_bat = self.var["E_bat"]

        # Objective: member deviation costs, PV costs, and individual import/export costs.
        m.setObjective(gp.quicksum((c.members[i].linear_disutility if c.members[i].linear_disutility is not None else 0.0) * (d_pos[i,t] + d_neg[i,t]) + c.members[i].pv_marginal_cost * g[i,t] + (c.members[i].energy_price[t] + c.members[i].import_tariff) * q_imp[i,t] - (c.members[i].energy_price[t] - c.members[i].export_tariff) * q_exp[i,t] for i in I for t in T), GRB.MINIMIZE)

        # Constraints
        # Deviations from reference load
        self.con["deviation"] = m.addConstrs(((l[i,t] - c.members[i].reference_load[t] ==  d_pos[i,t] - d_neg[i,t]) for i in I for t in T), name="deviation")

        # Load bounds
        self.con["load_lower_bounds"] = m.addConstrs((c.members[i].load_lower_kWh[t] <= l[i, t] for i in I for t in T), name="load_lower")
        self.con["load_upper_bounds"] = m.addConstrs((l[i, t] <= c.members[i].load_upper_kWh[t] for i in I for t in T), name="load_upper")

        # Energy minimum constraint
        self.con["energy_min"] = m.addConstrs((gp.quicksum(l[i,t] for t in T)>= c.members[i].min_daily_energy_kWh for i in I), name="energy_min")

        # PV constraint
        self.con["pv_limit"] = m.addConstrs((g[i,t] <= c.members[i].pv_available[t] for i in I for t in T), name="pv_limit")

        # Load balance 
        self.con["load_balance"] = m.addConstrs((l[i,t] == g[i,t] + q_imp[i,t] - q_exp[i,t] - s_ch[i,t] + s_dis[i,t] for i in I for t in T), name="load_balance")

        # Battery constraints
        battery_members = [i for i in I if c.members[i].has_battery]
        nonbattery_members = [i for i in I if not c.members[i].has_battery]

        # No battery means no charge/discharge and zero stored energy
        self.con["no_battery_charge"] = m.addConstrs((s_ch[i, t] == 0 for i in nonbattery_members for t in T),name="no_battery_charge")
        self.con["no_battery_discharge"] = m.addConstrs((s_dis[i, t] == 0 for i in nonbattery_members for t in T),name="no_battery_discharge")
        self.con["no_battery_soc"] = m.addConstrs((E_bat[i, t] == 0 for i in nonbattery_members for t in range(n_hours + 1)),name="no_battery_soc")
        
        #SOC constraints
        self.con["soc_initial"] = m.addConstrs((E_bat[i,0]== c.members[i].battery_initial_soc_kWh for i in battery_members), name="soc_initial")
        self.con["soc_dynamic"] = m.addConstrs((E_bat[i, t + 1] == E_bat[i, t] + c.members[i].battery_charging_efficiency * s_ch[i, t] - s_dis[i, t] / c.members[i].battery_discharging_efficiency for i in battery_members for t in T), name="soc_dynamic")

        #Energy bounds constraints
        #T: 0,..,23
        #n_hours+1 used bc battery has one more time index than the hourly decision variables
        self.con["soc_max"] = m.addConstrs((E_bat[i,t] <= c.members[i].battery_capacity_kWh for i in battery_members for t in range(n_hours + 1)), name="soc_max")
        self.con["charge_limit"] = m.addConstrs((s_ch[i,t] <= c.members[i].battery_max_charge_kW for i in battery_members for t in T),name="charge_limit")
        self.con["discharge_limit"] = m.addConstrs((s_dis[i,t] <= c.members[i].battery_max_discharge_kW for i in battery_members for t in T),name="discharge_limit")
        self.con["soc_final"] = m.addConstrs((E_bat[i, n_hours] >= (c.members[i].battery_final_soc_kWh if c.members[i].battery_final_soc_kWh is not None else END_OF_DAY_CHARGE) for i in battery_members),name="soc_final")
        
        return self
        
    def solve(self):

        self.m.optimize()

        if self.m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Model R did not reach optimality. Status: {self.m.Status}")
        validation = validate_solution(self.m)

        print("\n--- Model validation ---")
        for key, value in validation.items():
            if key == "validation_messages":
                for message in value:
                    print(f"- {message}")
            else:
                print(f"{key}: {value}")

        return {
            "model_name": "Model R",
            "status": self.m.Status,
            "objective": self.m.ObjVal,
            **validation,
        }



class ModelI:
    """Model I: individual imports, exports, and DSO caps."""

    def __init__(self, community: CommunityData, verbose: bool = False):
        self.community = community
        self.verbose = verbose
        self.I = range(len(community.members))
        self.T = range(len(community.energy_price))
        self.m = gp.Model("model_i")
        self.m.Params.OutputFlag = int(verbose)
        self.var = {}
        self.con = {}

    def build(self) -> "ModelI":
        m, c, I, T = self.m, self.community, self.I, self.T
        n_hours = len(T)

        # Consumer and Hourly variables
        self.var["l"] = m.addVars(I, T, lb=0, vtype=GRB.CONTINUOUS, name="l")
        self.var["g"] = m.addVars(I, T, lb=0, vtype=GRB.CONTINUOUS, name="g")
        self.var["q_imp"] = m.addVars(I, T, lb=0, vtype=GRB.CONTINUOUS, name="q_imp")
        self.var["q_exp"] = m.addVars(I, T, lb=0, vtype=GRB.CONTINUOUS, name="q_exp")
        self.var["d_+"] = m.addVars(I, T, lb=0, vtype=GRB.CONTINUOUS, name="d_pos")
        self.var["d_-"] = m.addVars(I, T, lb=0, vtype=GRB.CONTINUOUS, name="d_neg")

        # Model I has a separate DSO excess-import variable for every member and hour.
        self.var["e"] = m.addVars(I, T, lb=0, vtype=GRB.CONTINUOUS, name="dso_excess")

        # Battery charging and discharging are modelled hourly, with one extra SOC state for the end of the day.
        self.var["s_ch"] = m.addVars(I, T, lb=0, vtype=GRB.CONTINUOUS, name="s_ch")
        self.var["s_dis"] = m.addVars(I, T, lb=0, vtype=GRB.CONTINUOUS, name="s_dis")
        self.var["E_bat"] = m.addVars(I, range(n_hours + 1), lb=0, vtype=GRB.CONTINUOUS, name="E_bat")

        l, g, q_imp, q_exp = self.var["l"], self.var["g"], self.var["q_imp"], self.var["q_exp"]
        d_pos, d_neg, e = self.var["d_+"], self.var["d_-"], self.var["e"]
        s_ch, s_dis, E_bat = self.var["s_ch"], self.var["s_dis"], self.var["E_bat"]

        # DSO-adjusted import and export prices. The discount applies to the tariffs, not the day-ahead price.
        beta = c.contract.tariff_discount_ratio
        p_imp = {t: c.energy_price[t] + (1 - beta) * c.import_tariff for t in T}
        p_exp = {t: c.energy_price[t] - (1 - beta) * c.export_tariff for t in T}
        alpha = c.contract.penalty_DKK_per_kWh

        # NEW OBJECTIVE: each member pays for individual imports, receives revenue for individual exports, and pays their own deviation and PV costs.
        # Model I charges a penalty for each member's imports above that member's allocated DSO cap.
        m.setObjective(gp.quicksum(((c.members[i].linear_disutility if c.members[i].linear_disutility is not None else 0.0) * (d_pos[i, t] + d_neg[i, t]) + c.members[i].pv_marginal_cost * g[i, t] + p_imp[t] * q_imp[i, t] - p_exp[t] * q_exp[i, t] + alpha * e[i, t]) for i in I for t in T), GRB.MINIMIZE)

        # NEW MODEL I CONSTRAINT: each member has an individual DSO capacity allocation, kappa_i * P_DSO[t].
        # Excess is penalized only when that member's import exceeds their own allocated cap.
        self.con["dso_excess"] = m.addConstrs((e[i, t] >= q_imp[i, t] - c.contract.individual_capacity_limit_kW[c.members[i].consumer_id][t] for i in I for t in T), name="dso_excess")

        # Load, Energy, PV constraints (NOTHING NEW)
        self.con["deviation"] = m.addConstrs((l[i, t] - c.members[i].reference_load[t] == d_pos[i, t] - d_neg[i, t] for i in I for t in T), name="deviation")
        self.con["load_lower"] = m.addConstrs((l[i, t] >= c.members[i].load_lower_kWh[t] for i in I for t in T), name="load_lower")
        self.con["load_upper"] = m.addConstrs((l[i, t] <= c.members[i].load_upper_kWh[t] for i in I for t in T), name="load_upper")
        self.con["energy_min"] = m.addConstrs((gp.quicksum(l[i, t] for t in T) >= c.members[i].min_daily_energy_kWh for i in I), name="energy_min")
        self.con["pv_limit"] = m.addConstrs((g[i, t] <= c.members[i].pv_available[t] for i in I for t in T), name="pv_limit")
        self.con["load_balance"] = m.addConstrs((l[i, t] == g[i, t] + q_imp[i, t] - q_exp[i, t] - s_ch[i, t] + s_dis[i, t] for i in I for t in T), name="load_balance")

        # Battery member constraints (NOTHING NEW)
        battery_members = [i for i in I if c.members[i].has_battery]
        nonbattery_members = [i for i in I if not c.members[i].has_battery]
        self.con["no_battery_charge"] = m.addConstrs((s_ch[i, t] == 0 for i in nonbattery_members for t in T), name="no_battery_charge")
        self.con["no_battery_discharge"] = m.addConstrs((s_dis[i, t] == 0 for i in nonbattery_members for t in T), name="no_battery_discharge")
        self.con["no_battery_soc"] = m.addConstrs((E_bat[i, t] == 0 for i in nonbattery_members for t in range(n_hours + 1)), name="no_battery_soc")

        # Battery constraints (NOTHING NEW)
        self.con["soc_initial"] = m.addConstrs((E_bat[i, 0] == c.members[i].battery_initial_soc_kWh for i in battery_members), name="soc_initial")
        self.con["soc_dynamic"] = m.addConstrs((E_bat[i, t + 1] == E_bat[i, t] + c.members[i].battery_charging_efficiency * s_ch[i, t] - s_dis[i, t] / c.members[i].battery_discharging_efficiency for i in battery_members for t in T), name="soc_dynamic")
        self.con["soc_max"] = m.addConstrs((E_bat[i, t] <= c.members[i].battery_capacity_kWh for i in battery_members for t in range(n_hours + 1)), name="soc_max")
        self.con["charge_limit"] = m.addConstrs((s_ch[i, t] <= c.members[i].battery_max_charge_kW for i in battery_members for t in T), name="charge_limit")
        self.con["discharge_limit"] = m.addConstrs((s_dis[i, t] <= c.members[i].battery_max_discharge_kW for i in battery_members for t in T), name="discharge_limit")
        self.con["soc_final"] = m.addConstrs((E_bat[i, n_hours] >= (c.members[i].battery_final_soc_kWh if c.members[i].battery_final_soc_kWh is not None else END_OF_DAY_CHARGE) for i in battery_members), name="soc_final")

        return self

    def solve(self):
        """Optimize Model I and validate the solution."""
        self.m.optimize()
        if self.m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Model I did not reach optimality. Status: {self.m.Status}")

        validation = validate_solution(self.m)
        print("\n--- Model I validation ---")
        for key, value in validation.items():
            if key == "validation_messages":
                for message in value:
                    print(f"- {message}")
            else:
                print(f"{key}: {value}")

        return {
                    "model_name": "Model I",
                    "status": self.m.Status,
                    "objective": self.m.ObjVal,
                    **validation,
                }



class ModelC:
    """Assignment 2, Model C: community-level imports, exports, and DSO cap."""

    def __init__(self, community: CommunityData, verbose: bool = False):
        self.community = community
        self.verbose = verbose
        self.I = range(len(community.members))
        self.T = range(len(community.energy_price))
        self.m = gp.Model("model_c")
        self.m.Params.OutputFlag = int(verbose)
        self.var = {}
        self.con = {}

    def build(self) -> "ModelC":
        m, c, I, T = self.m, self.community, self.I, self.T
        n_hours = len(T)

        # Consumer and Hourly variables
        self.var["l"] = m.addVars(I, T, lb=0, vtype=GRB.CONTINUOUS, name="l")
        self.var["g"] = m.addVars(I, T, lb=0, vtype=GRB.CONTINUOUS, name="g")
        self.var["d_+"] = m.addVars(I, T, lb=0, vtype=GRB.CONTINUOUS, name="d_pos")
        self.var["d_-"] = m.addVars(I, T, lb=0, vtype=GRB.CONTINUOUS, name="d_neg")
        self.var["s_ch"] = m.addVars(I, T, lb=0, vtype=GRB.CONTINUOUS, name="s_ch")
        self.var["s_dis"] = m.addVars(I, T, lb=0, vtype=GRB.CONTINUOUS, name="s_dis")
        self.var["E_bat"] = m.addVars(I, range(n_hours + 1), lb=0, vtype=GRB.CONTINUOUS, name="E_bat")

        # NEW MODEL C VARIABLE: q[i,t] is each member's net consumption.
        # It can be negative when a member's net energy flow is an export.
        self.var["q"] = m.addVars(I, T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="q")

        # NEW MODEL C VARIABLES: imports and exports are measured at the community's single connection point.
        # These replace Model I's separate q_imp[i,t] and q_exp[i,t] variables.
        self.var["P_imp"] = m.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="P_imp")
        self.var["P_exp"] = m.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="P_exp")

        # NEW MODEL C VARIABLE: one excess-import variable per hour for the entire community.
        self.var["P_exc"] = m.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="P_exc")

        l, g, q = self.var["l"], self.var["g"], self.var["q"]
        P_imp, P_exp, P_exc = self.var["P_imp"], self.var["P_exp"], self.var["P_exc"]
        d_pos, d_neg = self.var["d_+"], self.var["d_-"]
        s_ch, s_dis, E_bat = self.var["s_ch"], self.var["s_dis"], self.var["E_bat"]

        # DSO-adjusted prices same as Model I.
        beta = c.contract.tariff_discount_ratio
        p_imp = {t: c.energy_price[t] + (1 - beta) * c.import_tariff for t in T}
        p_exp = {t: c.energy_price[t] - (1 - beta) * c.export_tariff for t in T}
        alpha = c.contract.penalty_DKK_per_kWh

        # NEW Objective: the community pays for aggregate imports and receives revenue for aggregate exports.
        # Unlike Model I, there is only one congestion penalty per hour, based on total community imports.
        community_cost = gp.quicksum((p_imp[t] * P_imp[t] - p_exp[t] * P_exp[t] + alpha * P_exc[t]) for t in T)
        # Individual load deviation and PV production costs are still paid for each member.
        member_cost = gp.quicksum((c.members[i].linear_disutility if c.members[i].linear_disutility is not None else 0.0) * (d_pos[i, t] + d_neg[i, t]) + c.members[i].pv_marginal_cost * g[i, t] for i in I for t in T)
        m.setObjective(community_cost + member_cost, GRB.MINIMIZE)

        # NEW MODEL C CONSTRAINT: the community's excess import is measured against the shared hourly DSO cap.
        # This differs from Model I, where each member has their own allocated cap.
        self.con["dso_excess"] = m.addConstrs((P_exc[t] >= P_imp[t] - c.contract.capacity_limit_kW[t] for t in T), name="dso_excess")

        # Individual load deviations and flexible-load bounds remain member-specific, just as in Model I.
        self.con["deviation"] = m.addConstrs((l[i, t] - c.members[i].reference_load[t] == d_pos[i, t] - d_neg[i, t] for i in I for t in T), name="deviation")
        self.con["load_lower"] = m.addConstrs((l[i, t] >= c.members[i].load_lower_kWh[t] for i in I for t in T), name="load_lower")
        self.con["load_upper"] = m.addConstrs((l[i, t] <= c.members[i].load_upper_kWh[t] for i in I for t in T), name="load_upper")
        self.con["energy_min"] = m.addConstrs((gp.quicksum(l[i, t] for t in T) >= c.members[i].min_daily_energy_kWh for i in I), name="energy_min")
        self.con["pv_limit"] = m.addConstrs((g[i, t] <= c.members[i].pv_available[t] for i in I for t in T), name="pv_limit")

        # Member-level energy balance uses net consumption q[i,t], rather than separate member imports and exports.
        self.con["load_balance"] = m.addConstrs((l[i, t] == g[i, t] + q[i, t] - s_ch[i, t] + s_dis[i, t] for i in I for t in T), name="load_balance")

        # NEW MODEL C CONSTRAINT: aggregate member net consumption equals community imports minus community exports.
        # This equation links all members to the single meter at the community connection point.
        self.con["community_balance"] = m.addConstrs((gp.quicksum(q[i, t] for i in I) == P_imp[t] - P_exp[t] for t in T), name="community_balance")

        # Battery member constraints (NOTHING NEW)
        battery_members = [i for i in I if c.members[i].has_battery]
        nonbattery_members = [i for i in I if not c.members[i].has_battery]
        self.con["no_battery_charge"] = m.addConstrs((s_ch[i, t] == 0 for i in nonbattery_members for t in T), name="no_battery_charge")
        self.con["no_battery_discharge"] = m.addConstrs((s_dis[i, t] == 0 for i in nonbattery_members for t in T), name="no_battery_discharge")
        self.con["no_battery_soc"] = m.addConstrs((E_bat[i, t] == 0 for i in nonbattery_members for t in range(n_hours + 1)), name="no_battery_soc")

        # Battery constraints (NOTHING NEW)
        self.con["soc_initial"] = m.addConstrs((E_bat[i, 0] == c.members[i].battery_initial_soc_kWh for i in battery_members), name="soc_initial")
        self.con["soc_dynamic"] = m.addConstrs((E_bat[i, t + 1] == E_bat[i, t] + c.members[i].battery_charging_efficiency * s_ch[i, t] - s_dis[i, t] / c.members[i].battery_discharging_efficiency for i in battery_members for t in T), name="soc_dynamic")
        self.con["soc_max"] = m.addConstrs((E_bat[i, t] <= c.members[i].battery_capacity_kWh for i in battery_members for t in range(n_hours + 1)), name="soc_max")
        self.con["charge_limit"] = m.addConstrs((s_ch[i, t] <= c.members[i].battery_max_charge_kW for i in battery_members for t in T), name="charge_limit")
        self.con["discharge_limit"] = m.addConstrs((s_dis[i, t] <= c.members[i].battery_max_discharge_kW for i in battery_members for t in T), name="discharge_limit")
        self.con["soc_final"] = m.addConstrs((E_bat[i, n_hours] >= (c.members[i].battery_final_soc_kWh if c.members[i].battery_final_soc_kWh is not None else END_OF_DAY_CHARGE) for i in battery_members), name="soc_final")

        return self

    def solve(self):
        """Optimize Model C and validate the solution."""
        self.m.optimize()
        if self.m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Model C did not reach optimality. Status: {self.m.Status}")

        validation = validate_solution(self.m)
        print("\n--- Model C validation ---")
        for key, value in validation.items():
            if key == "validation_messages":
                for message in value:
                    print(f"- {message}")
            else:
                print(f"{key}: {value}")

        return {
                    "model_name": "Model C",
                    "status": self.m.Status,
                    "objective": self.m.ObjVal,
                    **validation,
                }
