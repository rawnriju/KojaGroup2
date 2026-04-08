"""
evaluate_drl.py — Load the trained SAC model and run ONE clean full-year sim.

This is the run you point judges / output_calcs_comparison.ipynb at.

Inputs:
    models/best_model_sac/best_model.zip   (saved at end of train_drl.py)

Outputs (after this script):
    drl_output/eval/run_<N>/eplusout.csv   ← USE THIS in visualize_output.ipynb
    sac_eval_eplus_<timestamp>.csv           ← RL obs/action log (optional)

``drl_output/train/`` is only mid-training scratch; ignore for submission.
"""

import os

import pandas as pd
from stable_baselines3 import SAC

from train_drl import (
    USE_GRU_AND_FRAME_STACK,
    FRAME_STACK_N,
    N_OBS_FEATURES,
    _build_config,
    make_wrapped_env,
)


IDF_FILE      = os.path.join("..", "DOAS_wNeutralSupplyAir_wFanCoilUnits.idf")
WEATHER_FILE  = os.path.join("..", "FIN_TR_Tampere.Satakunnankatu.027440_TMYx.2004-2018.epw")
MODEL_PATH    = os.path.join("models", "best_model_sac", "best_model.zip")
EVAL_OUT      = "drl_output2/eval"


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
    stacked = USE_GRU_AND_FRAME_STACK and FRAME_STACK_N > 1

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
    df.to_csv(path, index=False)

    print()
    print("=" * 60)
    print(f"  EVALUATION COMPLETE — {len(df)} steps")
    print(f"  Policy:  GRU+stack={USE_GRU_AND_FRAME_STACK}  obs_dim={N_OBS_FEATURES}")
    print()
    print(f"  {path}")
    print(f"     └─ RL-side log")

    eval_dir = config["eplus_output_path"]
    run_dirs = sorted(
        [d for d in os.listdir(eval_dir)
         if os.path.isdir(os.path.join(eval_dir, d)) and d.startswith("run_")],
        key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0,
    ) if os.path.isdir(eval_dir) else []
    if run_dirs:
        latest = os.path.join(eval_dir, run_dirs[-1], "eplusout.csv")
        print(f"  {latest}")
        print(f"     └─ EnergyPlus → visualize_output.ipynb")
    print("=" * 60)

    return df


if __name__ == "__main__":
    eval_config = _build_config(IDF_FILE, WEATHER_FILE, EVAL_OUT, "test")
    evaluate(MODEL_PATH, eval_config, csv_prefix="sac_eval_eplus")
