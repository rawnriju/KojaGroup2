"""
train_drl.py — Template for training a DRL agent with behavioral cloning
                pre-training followed by SAC fine-tuning.

Pipeline:
    1. Define observation & action spaces (edit OBS_SPEC / ACTION_SPEC)
    2. Behavioral Cloning (BC) from expert trajectories
    3. Evaluate BC policy for one year
    4. Transfer BC weights into SAC and fine-tune

To customize the observation space, add/remove entries in OBS_SPEC.
To customize the action space, edit ACTION_SPEC.
All min/max arrays are built automatically from the spec dicts.
"""

import os
import json

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from imitation.algorithms import bc
from imitation.data.types import Transitions

from eplus_sim import EnergyPlusEnv


# =========================================================================
# 1. PATHS — edit these to match your local setup
# =========================================================================

IDF_FILE     = os.path.join("..", "DOAS_wNeutralSupplyAir_wFanCoilUnits.idf")
WEATHER_FILE = os.path.join("..", "FIN_TR_Tampere.Satakunnankatu.027440_TMYx.2004-2018.epw")
EXPERT_JSON  = "expert_data.json"
MODEL_DIR    = "models"
TRAIN_OUT    = "drl_output/train"
EVAL_OUT     = "drl_output/train_eval"

# Runtime toggles (can be set via environment variables for quick smoke tests)
FAST_MODE = os.getenv("DRL_FAST_MODE", "0") == "1"
BC_EPOCHS = int(os.getenv("DRL_BC_EPOCHS", "5" if FAST_MODE else "50"))
SAC_WARMUP_STEPS = int(os.getenv("DRL_SAC_WARMUP_STEPS", "200" if FAST_MODE else "5000"))
SAC_TRAIN_STEPS = int(os.getenv("DRL_SAC_TRAIN_STEPS", "2000" if FAST_MODE else "300000"))
EVAL_FREQ = int(os.getenv("DRL_EVAL_FREQ", "500" if FAST_MODE else "50000"))
RUN_BC_EVAL = os.getenv("DRL_RUN_BC_EVAL", "0" if FAST_MODE else "1") == "1"
SAC_LR = float(os.getenv("DRL_SAC_LR", "3e-4"))
SAC_BATCH = int(os.getenv("DRL_SAC_BATCH", "256"))
SAC_GAMMA = float(os.getenv("DRL_SAC_GAMMA", "0.99"))
SAC_TAU = float(os.getenv("DRL_SAC_TAU", "0.005"))
SAC_ENT = os.getenv("DRL_SAC_ENT", "auto")


# =========================================================================
# 2. OBSERVATION SPACE — alias: (min, max)
#    Add, remove, or reorder entries freely. The config arrays are built
#    automatically from this dict.
#    Aliases must match those defined in variable_config.py / eplus_sim.py.
# =========================================================================

OBS_SPEC = {
    # Environment
    "outdoor_temp":       (-25.0,      40.0),
    "plenum_temp":        (  0.0,      50.0),

    # Zone temperatures, humidity, CO2 (repeat per zone)
    "space1_temp":        ( 10.0,      35.0),
    "space1_rh":          (  0.0,     100.0),
    "space1_co2":         (400.0,    2000.0),

    "space2_temp":        ( 10.0,      35.0),
    "space2_rh":          (  0.0,     100.0),
    "space2_co2":         (400.0,    2000.0),

    "space3_temp":        ( 10.0,      35.0),
    "space3_rh":          (  0.0,     100.0),
    "space3_co2":         (400.0,    2000.0),

    "space4_temp":        ( 10.0,      35.0),
    "space4_rh":          (  0.0,     100.0),
    "space4_co2":         (400.0,    2000.0),

    "space5_temp":        ( 10.0,      35.0),
    "space5_rh":          (  0.0,     100.0),
    "space5_co2":         (400.0,    2000.0),

    # Energy meters (J per timestep)
    "electricity_hvac":   (      0.0,  50_000_000.0),
    "gas_total":          (      0.0, 100_000_000.0),

    # Time features
    "hour":               (  0.0,  23.0),
    "day_of_week":        (  1.0,   7.0),
}


# =========================================================================
# 3. ACTION SPACE — alias: (min, max)
#    Physical ranges; normalized to [-1, 1] when actuator_normalize=1.
# =========================================================================

ACTION_SPEC = {
    "cooling_setpoint":  (18.0, 25.0),     # °C
    "heating_setpoint":  (18.0, 25.0),     # °C
    "ahu_supply_temp":   (16.0, 21.0),     # °C
    "supply_fan_flow":   ( 0.0,  1.0),     # kg/s
}


# =========================================================================
# 4. SIMULATION SETTINGS
# =========================================================================

TIMESTEP_INTERVAL = 15                     # minutes per step
TOTAL_STEPS = int(
    os.getenv(
        "DRL_TOTAL_STEPS",
        str(96 * 14 + 576 if FAST_MODE else 96 * 365 + 576),
    )
)


# =========================================================================
# Build config dicts (auto-generated from specs above — do not edit)
# =========================================================================

def _build_config(idf, weather, output_path, phase):
    obs_names = list(OBS_SPEC.keys())
    act_names = list(ACTION_SPEC.keys())
    return {
        "eplus_idf_filename": idf,
        "weather_filename":   weather,
        "eplus_output_path":  output_path,
        "phase":              phase,
        "interval":           TIMESTEP_INTERVAL,
        "total_steps":        TOTAL_STEPS,

        "observations":         obs_names,
        "observation_normalize": 1,
        "rl_observation_min":   [OBS_SPEC[k][0] for k in obs_names],
        "rl_observation_max":   [OBS_SPEC[k][1] for k in obs_names],

        "rl_actions":           act_names,
        "actuator_normalize":   1,
        "action_range":         {k: list(v) for k, v in ACTION_SPEC.items()},
        "rl_action_min":        [-1.0] * len(act_names),
        "rl_action_max":        [ 1.0] * len(act_names),
    }

train_config = _build_config(IDF_FILE, WEATHER_FILE, TRAIN_OUT, "learn")
eval_config  = _build_config(IDF_FILE, WEATHER_FILE, EVAL_OUT,  "test")


# =========================================================================
# Helper: load expert trajectories
# =========================================================================

def load_expert_pairs(json_path, config):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    obs_keys = config["observations"]
    act_keys = config["rl_actions"]

    obs_list, act_list = [], []
    for row in data:
        obs_list.append([row["obs"][k] for k in obs_keys])
        act_list.append([row["action"][k] for k in act_keys])

    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32)


# =========================================================================
# Helper: run one full-year evaluation and save CSV
# =========================================================================

def evaluate_policy(policy, env, csv_prefix="bc_full_year_test"):
    obs_names = env.unwrapped.config["observations"]
    act_names = env.unwrapped.config["rl_actions"]

    obs, _ = env.reset()
    done, truncated = False, False
    rows = []

    while not (done or truncated):
        action, _ = policy.predict(obs, deterministic=True)
        next_obs, reward, done, truncated, _ = env.step(action)

        row = {name: float(obs[i]) for i, name in enumerate(obs_names)}
        row.update({name: float(action[i]) for i, name in enumerate(act_names)})
        row["reward"] = float(reward)
        row["done"] = done
        row["truncated"] = truncated
        rows.append(row)
        obs = next_obs

    df = pd.DataFrame(rows)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    path = f"{csv_prefix}_{ts}.csv"
    df.to_csv(path, index=False)
    print(f"Saved evaluation: {path}")
    return df


# =========================================================================
# Pipeline
# =========================================================================

if __name__ == "__main__":

    # --- Create environments ---
    train_env = Monitor(EnergyPlusEnv(train_config))
    eval_env  = Monitor(EnergyPlusEnv(eval_config))

    os.makedirs(MODEL_DIR, exist_ok=True)

    print(
        "Training configuration: "
        f"FAST_MODE={FAST_MODE}, TOTAL_STEPS={TOTAL_STEPS}, "
        f"BC_EPOCHS={BC_EPOCHS}, SAC_WARMUP_STEPS={SAC_WARMUP_STEPS}, "
        f"SAC_TRAIN_STEPS={SAC_TRAIN_STEPS}, EVAL_FREQ={EVAL_FREQ}, "
        f"LR={SAC_LR}, BATCH={SAC_BATCH}, GAMMA={SAC_GAMMA}, TAU={SAC_TAU}, ENT={SAC_ENT}"
    )

    # --- Step 1: Behavioral Cloning ---
    bc_trainer = None
    if os.path.isfile(EXPERT_JSON):
        obs, acts = load_expert_pairs(EXPERT_JSON, train_config)

        expert_data = Transitions(
            obs=obs,
            acts=acts,
            infos=np.array([{}] * len(obs), dtype=object),
            next_obs=np.zeros_like(obs),
            dones=np.zeros(len(obs), dtype=bool),
        )

        bc_policy = ActorCriticPolicy(
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            lr_schedule=lambda _: 1e-3,
            net_arch=[256, 256],
            activation_fn=torch.nn.ReLU,
        )

        bc_trainer = bc.BC(
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            demonstrations=expert_data,
            rng=np.random.default_rng(0),
            device="cpu",
            policy=bc_policy,
        )

        bc_trainer.train(n_epochs=BC_EPOCHS)
        bc_trainer.policy.save(os.path.join(MODEL_DIR, "bc_policy.pt"))
        print("BC training complete.")

        # --- Step 2: Evaluate BC policy ---
        if RUN_BC_EVAL:
            print("Evaluating BC policy...")
            evaluate_policy(bc_trainer.policy, eval_env, csv_prefix="bc_full_year_test")
            eval_env.close()
    else:
        print(f"WARNING: Expert data not found at '{EXPERT_JSON}'. Skipping BC and training SAC from scratch.")
        eval_env.close()

    # --- Step 3: SAC with BC weight transfer ---
    eval_env = Monitor(EnergyPlusEnv(eval_config))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_DIR, "best_model_sac"),
        log_path=os.path.join(MODEL_DIR, "eval_logs_sac"),
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False,
        n_eval_episodes=1,
    )

    sac_model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=SAC_LR,
        batch_size=SAC_BATCH,
        gamma=SAC_GAMMA,
        tau=SAC_TAU,
        ent_coef=SAC_ENT,
        verbose=1,
        tensorboard_log=os.path.join(MODEL_DIR, "tb_logs_sac"),
    )

    if bc_trainer is not None:
        # Transfer BC actor weights → SAC actor
        sac_model.policy.actor.latent_pi.load_state_dict(
            bc_trainer.policy.mlp_extractor.policy_net.state_dict()
        )
        sac_model.policy.actor.mu.load_state_dict(
            bc_trainer.policy.action_net.state_dict()
        )

        # Freeze actor for initial warm-up steps
        for p in sac_model.policy.actor.parameters():
            p.requires_grad = False
        sac_model.learn(total_timesteps=SAC_WARMUP_STEPS)

        for p in sac_model.policy.actor.parameters():
            p.requires_grad = True
    else:
        print("Running SAC without BC warm start.")

    # Full SAC training
    sac_model.learn(total_timesteps=SAC_TRAIN_STEPS, callback=eval_callback, progress_bar=True)

    sac_model.save(os.path.join(MODEL_DIR, "sac_bc_hvac"))
    print("SAC training complete.")