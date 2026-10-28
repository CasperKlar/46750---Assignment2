
import gurobipy as gp
from gurobipy import GRB


def validate_solution(model: gp.Model, tolerance: float = 1e-6) -> dict:
    """Validate a solved Gurobi model numerically."""

    messages = []
    is_optimal = model.Status == GRB.OPTIMAL

    if not is_optimal:
        messages.append(
            f"Model status is {model.Status}, not OPTIMAL."
        )
        return {
            "is_optimal": False,
            "is_feasible": False,
            "max_constraint_violation": float("inf"),
            "max_bound_violation": float("inf"),
            "objective_matches": False,
            "validation_passed": False,
            "validation_messages": messages,
        }

    # Gurobi provides these quality measures at model level.
    max_constraint_violation = model.getAttr(GRB.Attr.ConstrVio)
    max_bound_violation = model.getAttr(GRB.Attr.BoundVio)

    # Compare the evaluated objective with the reported objective.
    objective_from_solution = model.getObjective().getValue()
    objective_matches = abs(
        objective_from_solution - model.ObjVal
    ) <= tolerance * max(1.0, abs(model.ObjVal))

    is_feasible = (
        max_constraint_violation <= tolerance
        and max_bound_violation <= tolerance
    )

    if not is_feasible:
        messages.append(
            "Constraint or variable-bound violation detected."
        )

    if not objective_matches:
        messages.append("Objective consistency check failed.")

    if is_feasible and objective_matches:
        messages.append(
            "Numerical feasibility and objective checks passed."
        )

    validation_passed = (
        is_optimal and is_feasible and objective_matches
    )

    return {
        "is_optimal": is_optimal,
        "is_feasible": is_feasible,
        "max_constraint_violation": max_constraint_violation,
        "max_bound_violation": max_bound_violation,
        "objective_matches": objective_matches,
        "validation_passed": validation_passed,
        "validation_messages": messages,
    }
