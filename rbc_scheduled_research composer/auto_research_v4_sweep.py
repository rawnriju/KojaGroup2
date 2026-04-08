#!/usr/bin/env python3
"""Multi-trial auto-research: explore energy vs CO2/temp penalty trade-offs.

Runs EnergyPlus once per trial with merged PARAMS, logs JSONL, prints best total_cost_eur.

Usage:
  python auto_research_v4_sweep.py              # default curated + small random batch
  python auto_research_v4_sweep.py --random 12  # extra random trials
  python auto_research_v4_sweep.py --quick      # curated only (no random)
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_v3():
    path = SCRIPT_DIR / "auto_research_v3_pi.py"
    spec = importlib.util.spec_from_file_location("auto_research_v3_pi", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["auto_research_v3_pi"] = mod
    spec.loader.exec_module(mod)
    return mod


def merge_params(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    out.update(overrides)
    return out


def curated_trials():
    """Hand-designed directions: aggressive energy, comfort-heavy, economizer, CO2 slack, hours."""
    return [
        ("baseline_v3", {}),
        # Tighter to S1: expect lower temp penalty, higher energy
        ("comfort_htg_clg", {"s1_htg_margin": 0.62, "s1_clg_margin": 0.02, "target_offset": 0.45}),
        # Looser heating / allow warmer cooling band → less HVAC, may raise temp penalty
        ("aggressive_margins", {"s1_htg_margin": 0.38, "s1_clg_margin": 0.0, "target_offset": 0.55}),
        # Shorter occupied ventilation
        ("early_close_19", {"working_hours_end": 19.0}),
        ("early_close_18", {"working_hours_end": 18.0}),
        # CO2 DCV slack: fewer fan hours, risk CO2 euros
        ("co2_wide_620_800", {"co2_min_limit": 620, "co2_max_limit": 800}),
        ("co2_wide_630_820", {"co2_min_limit": 630, "co2_max_limit": 820}),
        ("co2_tight_680_740", {"co2_min_limit": 680, "co2_max_limit": 740}),
        # Lower minimum flow during ramp
        ("flow_low_03", {"flow_low": 0.03}),
        ("flow_low_04", {"flow_low": 0.04}),
        # Supply air: slightly warmer → less cooling plant (may affect comfort)
        ("supply_warmer", {"sup_temp_at_low": 17.4, "sup_temp_at_high": 16.4}),
        ("supply_cooler", {"sup_temp_at_low": 16.6, "sup_temp_at_high": 15.6}),
        # Economizer boost (new knob in v3)
        ("econ_boost_02", {"economizer_boost_flow": 0.2, "economizer_hot_margin": 0.9}),
        ("econ_boost_03_loose", {"economizer_boost_flow": 0.3, "economizer_hot_margin": 1.2}),
        ("econ_boost_025_tight", {"economizer_boost_flow": 0.25, "economizer_hot_margin": 0.65}),
        # Cap max fan (saves fan/coils if CO2 stays acceptable)
        ("flow_boost_085", {"flow_boost": 0.85}),
        ("flow_boost_09", {"flow_boost": 0.9}),
        # Combined hypothesis: a little temp slack + shorter day + mild economizer
        ("combo_slack_econ", {
            "s1_htg_margin": 0.42,
            "s1_clg_margin": 0.0,
            "working_hours_end": 19.0,
            "economizer_boost_flow": 0.22,
            "economizer_hot_margin": 0.85,
        }),
        # Push penalties slightly for energy (user idea)
        ("penalty_trade_push", {
            "s1_htg_margin": 0.32,
            "co2_min_limit": 640,
            "co2_max_limit": 790,
            "working_hours_end": 18.5,
            "flow_low": 0.035,
        }),
    ]


def random_overrides(rng: random.Random) -> dict:
    """Single random trial in a bounded box around typical ranges."""
    cmin = int(rng.choice([600, 620, 630, 650, 670, 680]))
    cmax = int(rng.choice([740, 760, 780, 800, 820]))
    if cmin >= cmax:
        cmax = cmin + 100
    sl = round(rng.uniform(16.6, 17.8), 2)
    sh = round(rng.uniform(15.4, 16.8), 2)
    if sh > sl - 0.25:
        sh = round(sl - 0.4, 2)
    return {
        "s1_htg_margin": round(rng.uniform(0.32, 0.62), 3),
        "s1_clg_margin": round(rng.uniform(0.0, 0.12), 3),
        "target_offset": round(rng.uniform(0.35, 0.65), 3),
        "co2_min_limit": cmin,
        "co2_max_limit": cmax,
        "working_hours_end": round(rng.uniform(17.5, 21.0), 2),
        "flow_low": round(rng.uniform(0.03, 0.07), 3),
        "flow_boost": round(rng.uniform(0.85, 1.0), 3),
        "sup_temp_at_low": sl,
        "sup_temp_at_high": sh,
        "economizer_boost_flow": rng.choice([0.0, 0.0, 0.15, 0.2, 0.28]),
        "economizer_hot_margin": round(rng.uniform(0.65, 1.15), 2),
    }


def main():
    ap = argparse.ArgumentParser(description="Sweep RBC params via EnergyPlus")
    ap.add_argument("--random", type=int, default=8, help="Number of random trials after curated")
    ap.add_argument("--quick", action="store_true", help="Only curated trials")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--append-log",
        action="store_true",
        help="Append to JSONL instead of truncating at start",
    )
    args = ap.parse_args()

    v3 = _load_v3()
    base = v3.PARAMS
    run_simulation = v3.run_simulation

    log_path = SCRIPT_DIR / "auto_research_v4_results.jsonl"
    if not args.append_log and log_path.exists():
        log_path.unlink()
    rng = random.Random(args.seed)

    trials = list(curated_trials())
    if not args.quick and args.random > 0:
        for i in range(args.random):
            trials.append((f"random_{args.seed}_{i+1}", random_overrides(rng)))

    runs_root = SCRIPT_DIR / "eplus_runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    def trial_out_dir(trial_name: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", trial_name).strip("_")[:96]
        return runs_root / safe

    results = []
    t0 = time.time()
    for name, ov in trials:
        params = merge_params(base, ov)
        print(f"\n>>> Trial {name!r} overrides={ov}")
        t_run = time.time()
        costs = run_simulation(params, output_dir=trial_out_dir(name))
        dt = time.time() - t_run
        row = {
            "trial": name,
            "overrides": ov,
            "elapsed_s": round(dt, 2),
            "costs": costs,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        results.append(row)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        if costs:
            print(
                f"    total={costs['total_cost_eur']:.2f} "
                f"(E={costs['energy_cost_eur']:.2f} CO2={costs['co2_penalty_eur']:.2f} "
                f"T={costs['temp_penalty_eur']:.2f}) [{dt:.1f}s]"
            )
        else:
            print("    FAILED")

    ok = [r for r in results if r.get("costs")]
    if not ok:
        print("No successful runs.")
        return 1

    best = min(ok, key=lambda r: r["costs"]["total_cost_eur"])
    print(f"\n{'='*60}")
    print(f"Sweep done in {time.time()-t0:.0f}s | log: {log_path}")
    print(f"BEST: {best['trial']} → TOTAL {best['costs']['total_cost_eur']:.2f} EUR")
    print(f"  energy={best['costs']['energy_cost_eur']:.2f} co2={best['costs']['co2_penalty_eur']:.2f} "
          f"temp={best['costs']['temp_penalty_eur']:.2f}")
    print(f"  overrides: {best['overrides']}")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
