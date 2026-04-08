"""
evaluate_drl.py — Load a trained model and run a full-year evaluation.

Uses OBS_SPEC / ACTION_SPEC from train_drl so that the observation and
action spaces stay in sync.  Wraps the environment with FrameStackWrapper
to match the training-time observation format.
"""

import os

import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from eplus_sim import EnergyPlusEnv, FrameStackWrapper
from train_drl import OBS_SPEC, ACTION_SPEC, FRAME_STACK_N, _build_config


# =========================================================================
# PATHS
# =========================================================================

IDF_FILE      = os.path.join("..", "DOAS_wNeutralSupplyAir_wFanCoilUnits.idf")
WEATHER_FILE  = os.path.join("..", "FIN_TR_Tampere.Satakunnankatu.027440_TMYx.2004-2018.epw")
MODEL_PATH    = os.path.join("models", "best_model_sac", "best_model.zip")
EVAL_OUT      = "drl_output/eval"


# =========================================================================
# Evaluation
# =========================================================================

def evaluate(model_path, config, csv_prefix="eval"):
    base_env = EnergyPlusEnv(config)
    env = Monitor(FrameStackWrapper(base_env, n_frames=FRAME_STACK_N))
    model = SAC.load(model_path, env=env)

    obs_names = config["observations"]
    act_names = config["rl_actions"]

    obs, _ = env.reset()
    done, truncated = False, False
    rows = []

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, truncated, _ = env.step(action)

        n_base = len(obs_names)
        base_obs = obs[-n_base:] if len(obs) > n_base else obs
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
    print(f"Saved evaluation ({len(df)} steps): {path}")
    return df


if __name__ == "__main__":
    eval_config = _build_config(IDF_FILE, WEATHER_FILE, EVAL_OUT, "test")
    evaluate(MODEL_PATH, eval_config, csv_prefix="sac_eval")
