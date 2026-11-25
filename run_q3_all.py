"""Run all first Q3 results and save everything.

    python run_q3_all.py            # run everything; runs that already finished are skipped
    python run_q3_all.py --force    # run everything again

Outputs (in results/Q3/base/d1/):
    run_logs/<name>.txt      the full printout of every command
    q3_first_results.csv     one row per model: the table for the report
"""
import os, re, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.Q3_results import summarize

SAVE = ROOT / "results" / "Q3" / "base" / "d1"
LOGS = SAVE / "run_logs"
LOGS.mkdir(parents=True, exist_ok=True)
FORCE = "--force" in sys.argv
EPS = "0.2"

# name, N, command arguments for main.py
RUNS = [
    ("RN_N5",   5,  ["--Q3", "sdp_rn",   "--N", "5",  "--req", "BB"]),
    ("RN_N10",  10, ["--Q3", "sdp_rn",   "--N", "10", "--req", "BB"]),
    ("CVaR_N5", 5,  ["--Q3", "sdp_cvar", "--N", "5",  "--req", "BB"]),
    ("CVaR_N10", 10, ["--Q3", "sdp_cvar", "--N", "10", "--req", "BB"]),
    ("CC_N5",   5,  ["--Q3", "sdp_cc",   "--N", "5",  "--req", "BB", "--epsilon", EPS]),
    ("CC_N10",  10, ["--Q3", "sdp_cc",   "--N", "10", "--req", "BB", "--epsilon", EPS]),
]
BASELINES = [("DPBB", "DPBB")]   # name, tag


def run_main(args, log):
    """Run main.py with the given arguments; the whole printout goes to the log file."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, "-u", str(ROOT / "main.py")] + args
    with open(log, "w", encoding="utf-8") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
    return log.read_text(encoding="utf-8", errors="ignore")


def grab(text, pattern):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else np.nan


def find_tag(text):
    m = re.search(r"tag for the oos step: (\S+)", text)
    return m.group(1) if m else None


def oos_row(tag):
    """Test one price vector on the 100 unseen scenarios and return the summary numbers."""
    csv = SAVE / f"oos_{tag}.csv"
    if FORCE or not csv.exists():
        run_main(["--Q3", "oos", "--tag", tag], LOGS / f"oos_{tag}.txt")
    df = pd.read_csv(csv)
    return {"oos " + k: v for k, v in summarize(df).items()}


rows = []

def save_table():
    pd.DataFrame(rows).to_csv(SAVE / "q3_first_results.csv", index=False)


# ---------- part 1: the stochastic models choose prices, then get tested out-of-sample ----------
for name, N, args in RUNS:
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    log = LOGS / f"{name}.txt"
    text = log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""
    tag = find_tag(text)
    if FORCE or tag is None or not (SAVE / f"sdp_scenarios_{tag}.csv").exists():
        print("running (this can take up to 10 minutes)...", flush=True)
        text = run_main(args, log)
        tag = find_tag(text)
    else:
        print("already finished, reusing the saved log")

    row = {"model": name, "N": N, "tag": tag}
    if tag is None:
        row["note"] = "NO SOLUTION (see run_logs)"
        print("NO SOLUTION, see", log)
        rows.append(row); save_table()
        continue

    # solver numbers: printed by main.py
    row["gap_pct"] = grab(text, r"[Gg]ap:? ([\d.]+)%")
    row["runtime_s"] = grab(text, r"[Rr]untime:? ([\d.]+)\s*s")
    row["binaries"] = grab(text, r"binaries (\d+)")

    # in-sample numbers: computed from the saved per-scenario file (all scenarios have probability 1/N)
    sc = pd.read_csv(SAVE / f"sdp_scenarios_{tag}.csv")
    cost = sc["cost"] if "cost" in sc.columns else sc["cost_DKK"]
    bud = sc["budget"] if "budget" in sc.columns else sc["budget_DKK"]
    exc = sc["cap_exceedance_kWh"]
    row["in E[C] [DKK]"] = cost.mean()
    row["in worst scenario cost [DKK]"] = cost.max()
    row["in mean budget residual [DKK]"] = bud.mean()
    row["in scenarios above cap"] = int((exc > 1e-6).sum())
    row["in P(cap exceeded)"] = float((exc > 1e-6).mean())
    row["in max exceedance [kWh]"] = exc.max()

    print("testing on the 100 unseen scenarios...", flush=True)
    row.update(oos_row(tag))
    row["optimism gap [DKK]"] = row["oos mean community cost [DKK]"] - row["in E[C] [DKK]"]
    rows.append(row); save_table()
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)

# ---------- part 2: the baselines (prices come from Q2) ----------
for name, tag in BASELINES:
    print(f"\n=== baseline {name} ===", flush=True)
    row = {"model": name, "N": "-", "tag": tag}
    try:
        row.update(oos_row(tag))
    except Exception as e:
        row["note"] = f"failed: {e}"
        print("failed:", e)
    rows.append(row); save_table()

print("\nAll done. Table saved to", SAVE / "q3_first_results.csv")
print(pd.DataFrame(rows).T.to_string())
