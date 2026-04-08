"""
evaluate_transformer.py — Load Transformer+SAC policy; one full-year sim.

Same RL log columns as drl/evaluate_drl.py:
    <observation keys> + <action keys> + reward, done, truncated

EnergyPlus CSV:
    drl_transformer_output/eval/run_<N>/eplusout.csv
"""

import os
import sys

import pandas as pd
from stable_baselines3 import SAC

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
_DRL = os.path.join(_REPO, "drl")
if _DRL not in sys.path:
    sys.path.insert(0, _DRL)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import train_drl as td  # noqa: E402

from train_transformer import (  # noqa: E402
    FRAME_STACK_N,
    N_OBS_FEATURES,
    make_wrapped_env,
)

IDF_FILE = os.path.join(_REPO, "DOAS_wNeutralSupplyAir_wFanCoilUnits.idf")
WEATHER_FILE = os.path.join(
    _REPO, "FIN_TR_Tampere.Satakunnankatu.027440_TMYx.2004-2018.epw"
)
MODEL_PATH = os.path.join(_HERE, "models", "best_model_sac", "best_model.zip")
EVAL_OUT = os.path.abspath(os.path.join(_HERE, "drl_transformer_output", "eval"))


def evaluate(model_path, config, csv_prefix="sac_eval_eplus"):
    print("Building env...", flush=True)
    env = make_wrapped_env(config)
    print("Loading model...", flush=True)
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    model = SAC.load(model_path, env=env, device="cpu", custom_objects=custom_objects)
    print("Model loaded.", flush=True)

    obs_names = config["observations"]
    act_names = config["rl_actions"]
    n_base = len(obs_names)
    stacked = FRAME_STACK_N > 1

    print("Resetting env...")
    obs, _ = env.reset()
    print("Env reset complete.")
    done, truncated = False, False
    rows = []

    print("Starting evaluation loop...")
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, truncated, _ = env.step(action)

        base_obs = obs[-n_base:] if stacked else obs
        row = {name: float(base_obs[i]) for i, name in enumerate(obs_names)}
        row.update({name: float(action[i]) for i, name in enumerate(act_names)})
        row["reward"] = float(reward)
        row["done"] = done
        row["truncated"] = truncated
        rows.append(row)
        obs = next_obs

    env.close()

    df = pd.DataFrame(rows)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    path = f"{csv_prefix}_{ts}.csv"
    # Write next to script (same pattern as drl/evaluate_drl.py cwd-relative name)
    out_csv = os.path.join(_HERE, path)
    df.to_csv(out_csv, index=False)

    print()
    print("=" * 60)
    print(f"  EVALUATION COMPLETE — {len(df)} steps")
    print(f"  Policy: Transformer  frame_stack={FRAME_STACK_N}  obs_dim={N_OBS_FEATURES}")
    print()
    print(f"  {out_csv}")
    print(f"     └─ RL-side log")

    eval_dir = config["eplus_output_path"]
    run_dirs = (
        sorted(
            [
                d
                for d in os.listdir(eval_dir)
                if os.path.isdir(os.path.join(eval_dir, d)) and d.startswith("run_")
            ],
            key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0,
        )
        if os.path.isdir(eval_dir)
        else []
    )
    if run_dirs:
        latest = os.path.join(eval_dir, run_dirs[-1], "eplusout.csv")
        print(f"  {latest}")
        print(f"     └─ EnergyPlus → visualize_output.ipynb")
    print("=" * 60)

    return df


if __name__ == "__main__":
    eval_config = td._build_config(IDF_FILE, WEATHER_FILE, EVAL_OUT, "test")
    evaluate(MODEL_PATH, eval_config, csv_prefix="sac_eval_eplus")
