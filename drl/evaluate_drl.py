"""
evaluate_drl.py — Load the trained SAC model and run a full-year evaluation.

Uses the same env wrapper as train_drl.py (see USE_GRU_AND_FRAME_STACK there).

Outputs:
    sac_eval_eplus_TIMESTAMP.csv     — RL log (normalised obs, actions, rewards)
    drl_output/eval/run_N/eplusout.csv — for visualize_output.ipynb
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
EVAL_OUT      = "drl_output/eval"


def evaluate(model_path, config, csv_prefix="sac_eval_eplus"):
    env = make_wrapped_env(config)
    model = SAC.load(model_path, env=env)

    obs_names = config["observations"]
    act_names = config["rl_actions"]
    n_base = len(obs_names)
    stacked = USE_GRU_AND_FRAME_STACK and FRAME_STACK_N > 1

    obs, _ = env.reset()
    done, truncated = False, False
    rows = []

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
