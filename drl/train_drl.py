"""
train_drl.py — DRL agent: behavioral cloning + SAC fine-tuning.

Speed vs quality (default = fast, ~2×+ faster than 300 K steps + GRU):
    • Wall time is dominated by EnergyPlus (one HVAC timestep per env.step).
    • ``SAC_TOTAL_STEPS`` scales linearly with runtime — halve it ≈ halve train time.
    • GRU + 4-frame stack added heavy PyTorch work each step; default is plain MLP.

To train longer / richer policy, edit the constants in section 4:
    USE_GRU_AND_FRAME_STACK = True
    FRAME_STACK_N = 4
    SAC_TOTAL_STEPS = 300_000
    SAC_FROZEN_STEPS = 5_000

After training:
    python evaluate_drl.py

Expert data (v3 scheduler):
    python generate_expert_best_v1.py  →  expert_data_best_v3.json
"""

import os
import json
import math
from collections import deque

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from imitation.algorithms import bc
from imitation.data.types import Transitions

from eplus_sim import EnergyPlusEnv, FrameStackWrapper
from feature_extractors import GRUFeatureExtractor


# =========================================================================
# 1. PATHS
# =========================================================================

IDF_FILE     = os.path.join("..", "DOAS_wNeutralSupplyAir_wFanCoilUnits.idf")
WEATHER_FILE = os.path.join("..", "FIN_TR_Tampere.Satakunnankatu.027440_TMYx.2004-2018.epw")
EXPERT_JSON  = os.path.join(os.path.dirname(__file__), "expert_data_best_v3.json")
MODEL_DIR    = "models"
TRAIN_OUT    = "drl_output/train"


# =========================================================================
# 2. OBSERVATION SPACE (31 features)
# =========================================================================

OBS_SPEC = {
    "outdoor_temp":       (-25.0,      40.0),
    "plenum_temp":        (  0.0,      50.0),

    "space1_temp":        ( 10.0,      35.0),
    "space1_rh":          (  0.0,     100.0),
    "space1_co2":         (400.0,    2000.0),
    "space1_occ":         (  0.0,      50.0),

    "space2_temp":        ( 10.0,      35.0),
    "space2_rh":          (  0.0,     100.0),
    "space2_co2":         (400.0,    2000.0),
    "space2_occ":         (  0.0,      50.0),

    "space3_temp":        ( 10.0,      35.0),
    "space3_rh":          (  0.0,     100.0),
    "space3_co2":         (400.0,    2000.0),
    "space3_occ":         (  0.0,      50.0),

    "space4_temp":        ( 10.0,      35.0),
    "space4_rh":          (  0.0,     100.0),
    "space4_co2":         (400.0,    2000.0),
    "space4_occ":         (  0.0,      50.0),

    "space5_temp":        ( 10.0,      35.0),
    "space5_rh":          (  0.0,     100.0),
    "space5_co2":         (400.0,    2000.0),
    "space5_occ":         (  0.0,      50.0),

    "electricity_hvac":   (      0.0,  50_000_000.0),
    "gas_total":          (      0.0, 100_000_000.0),

    "hour_sin":           (-1.0,  1.0),
    "hour_cos":           (-1.0,  1.0),
    "day_sin":            (-1.0,  1.0),
    "day_cos":            (-1.0,  1.0),
    "month_sin":          (-1.0,  1.0),
    "month_cos":          (-1.0,  1.0),

    "outdoor_temp_rolling_24h": (-25.0, 40.0),
}

N_OBS_FEATURES = len(OBS_SPEC)


# =========================================================================
# 3. ACTION SPACE
# =========================================================================

ACTION_SPEC = {
    "cooling_setpoint":  (18.0, 25.0),
    "heating_setpoint":  (18.0, 25.0),
    "ahu_supply_temp":   (16.0, 21.0),
    "supply_fan_flow":   ( 0.0,  1.0),
}


# =========================================================================
# 4. SPEED / QUALITY KNOBS
#
#    Default: ~2×+ faster than (300 K + 5 K frozen + GRU + 4-frame stack)
#      — half as many EnergyPlus steps
#      — plain MLP (no GRU), no frame stacking
#      — SAC updates every 2 env steps (less GPU work; E+ still every step)
# =========================================================================

TIMESTEP_INTERVAL = 15
TOTAL_STEPS       = 96 * 365 + 576

USE_GRU_AND_FRAME_STACK = False   # True → slower, may help quality
FRAME_STACK_N     = 4             # only used if USE_GRU_AND_FRAME_STACK
GRU_HIDDEN        = 128
GRU_LAYERS        = 2

BC_EPOCHS         = 120
BC_LR             = 5e-4

SAC_FROZEN_STEPS  = 3_000
SAC_TOTAL_STEPS   = 150_000       # ~½ of 300 K → ~½ E+ time during SAC
SAC_BATCH_SIZE    = 256
SAC_BUFFER_SIZE   = 200_000
SAC_LEARNING_RATE = 3e-4
SAC_TRAIN_FREQ    = 2             # gradient update every N steps (1 = every step)


def _effective_frame_stack() -> int:
    return FRAME_STACK_N if USE_GRU_AND_FRAME_STACK else 1


def _wrap_env(base):
    n = _effective_frame_stack()
    if n > 1:
        return Monitor(FrameStackWrapper(base, n_frames=n))
    return Monitor(base)


def make_wrapped_env(config):
    """Used by evaluate_drl.py — must match training wrapper."""
    return _wrap_env(EnergyPlusEnv(config))


def _policy_kwargs():
    if USE_GRU_AND_FRAME_STACK:
        return {
            "features_extractor_class": GRUFeatureExtractor,
            "features_extractor_kwargs": {
                "n_frames": FRAME_STACK_N,
                "n_features_per_frame": N_OBS_FEATURES,
                "gru_hidden_size": GRU_HIDDEN,
                "gru_layers": GRU_LAYERS,
            },
            "net_arch": [256, 256],
            "share_features_extractor": False,
        }
    return {"net_arch": [256, 256]}


def _transfer_bc_to_sac(bc_policy, sac_model):
    if USE_GRU_AND_FRAME_STACK:
        sac_model.policy.actor.features_extractor.load_state_dict(
            bc_policy.features_extractor.state_dict())
        sac_model.policy.actor.latent_pi.load_state_dict(
            bc_policy.mlp_extractor.policy_net.state_dict())
        sac_model.policy.actor.mu.load_state_dict(bc_policy.action_net.state_dict())
        for net in (sac_model.policy.critic, sac_model.policy.critic_target):
            net.features_extractor.load_state_dict(bc_policy.features_extractor.state_dict())
    else:
        sac_model.policy.actor.latent_pi.load_state_dict(
            bc_policy.mlp_extractor.policy_net.state_dict())
        sac_model.policy.actor.mu.load_state_dict(bc_policy.action_net.state_dict())


# =========================================================================
# Build config dicts
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


# =========================================================================
# Helpers
# =========================================================================

def linear_schedule(initial_lr: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_lr
    return func


def _compute_derived_features(raw_obs: dict) -> dict:
    hour  = raw_obs.get("_raw_hour", 0.0)
    day   = raw_obs.get("_raw_day", 1.0)
    month = raw_obs.get("_raw_month", 1.0)

    raw_obs["hour_sin"]  = math.sin(2.0 * math.pi * hour / 24.0)
    raw_obs["hour_cos"]  = math.cos(2.0 * math.pi * hour / 24.0)
    raw_obs["day_sin"]   = math.sin(2.0 * math.pi * (day - 1) / 7.0)
    raw_obs["day_cos"]   = math.cos(2.0 * math.pi * (day - 1) / 7.0)
    raw_obs["month_sin"] = math.sin(2.0 * math.pi * (month - 1) / 12.0)
    raw_obs["month_cos"] = math.cos(2.0 * math.pi * (month - 1) / 12.0)
    return raw_obs


def load_expert_pairs(json_path, config, n_frames=None):
    if n_frames is None:
        n_frames = _effective_frame_stack()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    obs_keys = config["observations"]
    act_keys = config["rl_actions"]
    obs_min  = config["rl_observation_min"]
    obs_max  = config["rl_observation_max"]

    def normalise(obs_dict):
        vec = []
        for i, k in enumerate(obs_keys):
            v = float(obs_dict.get(k, 0.0))
            lo, hi = obs_min[i], obs_max[i]
            if hi > lo:
                v = 2.0 * (max(lo, min(hi, v)) - lo) / (hi - lo) - 1.0
            else:
                v = 0.0
            vec.append(v)
        return vec

    raw_outdoor_temps = []
    obs_dicts = []
    for row in data:
        obs_d = dict(row["obs"])
        if "hour_sin" not in obs_d and "_raw_hour" in obs_d:
            _compute_derived_features(obs_d)
        raw_outdoor_temps.append(obs_d.get("_raw_outdoor_temp",
                                           obs_d.get("outdoor_temp", 0.0)))
        obs_dicts.append(obs_d)

    outdoor_buf: deque = deque(maxlen=96)
    for i, obs_d in enumerate(obs_dicts):
        outdoor_buf.append(raw_outdoor_temps[i])
        obs_d["outdoor_temp_rolling_24h"] = sum(outdoor_buf) / len(outdoor_buf)

    obs_list = [normalise(d) for d in obs_dicts]
    act_list = [[row["action"][k] for k in act_keys] for row in data]

    obs  = np.array(obs_list, dtype=np.float32)
    acts = np.array(act_list, dtype=np.float32)

    if n_frames > 1:
        stacked = []
        buf: deque = deque(maxlen=n_frames)
        zero = np.zeros(obs.shape[1], dtype=np.float32)
        for _ in range(n_frames):
            buf.append(zero)
        for o in obs:
            buf.append(o)
            stacked.append(np.concatenate(list(buf)))
        obs = np.array(stacked, dtype=np.float32)

    return obs, acts


# =========================================================================
# Pipeline
# =========================================================================

if __name__ == "__main__":

    import time as _time
    _t0 = _time.time()

    train_env = make_wrapped_env(train_config)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.join(MODEL_DIR, "sac_final"), exist_ok=True)

    nf = _effective_frame_stack()
    print("=" * 60)
    print("  DRL training config")
    print(f"    GRU + frame stack: {USE_GRU_AND_FRAME_STACK}  (n_frames={nf})")
    print(f"    SAC frozen steps:  {SAC_FROZEN_STEPS:,}")
    print(f"    SAC train steps:   {SAC_TOTAL_STEPS:,}")
    print(f"    train_freq:        {SAC_TRAIN_FREQ}  (gradient update cadence)")
    print("=" * 60)

    print()
    print("  STEP 1 / 2 — Behavioral Cloning")

    obs, acts = load_expert_pairs(EXPERT_JSON, train_config, n_frames=nf)
    print(f"  Loaded {len(obs)} transitions  obs {obs.shape}  acts {acts.shape}")

    expert_data = Transitions(
        obs=obs,
        acts=acts,
        infos=np.array([{}] * len(obs), dtype=object),
        next_obs=np.zeros_like(obs),
        dones=np.zeros(len(obs), dtype=bool),
    )

    pk = _policy_kwargs()
    if USE_GRU_AND_FRAME_STACK:
        bc_policy = ActorCriticPolicy(
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            lr_schedule=lambda _: BC_LR,
            net_arch=pk["net_arch"],
            activation_fn=torch.nn.ReLU,
            features_extractor_class=pk["features_extractor_class"],
            features_extractor_kwargs=pk["features_extractor_kwargs"],
        )
    else:
        bc_policy = ActorCriticPolicy(
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            lr_schedule=lambda _: BC_LR,
            net_arch=pk["net_arch"],
            activation_fn=torch.nn.ReLU,
        )

    bc_trainer = bc.BC(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        demonstrations=expert_data,
        rng=np.random.default_rng(42),
        device="auto",
        policy=bc_policy,
        batch_size=SAC_BATCH_SIZE,
    )

    bc_trainer.train(n_epochs=BC_EPOCHS)
    bc_trainer.policy.save(os.path.join(MODEL_DIR, "bc_policy.pt"))
    print(f"  BC done  ({_time.time() - _t0:.0f}s)")

    print()
    print("  STEP 2 / 2 — SAC fine-tuning (EnergyPlus)")

    sac_model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=linear_schedule(SAC_LEARNING_RATE),
        batch_size=SAC_BATCH_SIZE,
        buffer_size=SAC_BUFFER_SIZE,
        learning_starts=500,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        target_entropy="auto",
        train_freq=SAC_TRAIN_FREQ,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=os.path.join(MODEL_DIR, "tb_logs_sac"),
        policy_kwargs=pk,
    )

    _transfer_bc_to_sac(bc_trainer.policy, sac_model)

    for p in sac_model.policy.actor.parameters():
        p.requires_grad = False
    sac_model.learn(total_timesteps=SAC_FROZEN_STEPS)

    for p in sac_model.policy.actor.parameters():
        p.requires_grad = True
    sac_model.learn(
        total_timesteps=SAC_TOTAL_STEPS,
        progress_bar=True,
        reset_num_timesteps=False,
    )

    final_path = os.path.join(MODEL_DIR, "sac_final", "sac_final")
    sac_model.save(final_path)
    best_dir = os.path.join(MODEL_DIR, "best_model_sac")
    os.makedirs(best_dir, exist_ok=True)
    sac_model.save(os.path.join(best_dir, "best_model"))

    elapsed = _time.time() - _t0
    print()
    print("=" * 60)
    print(f"  DONE — {elapsed / 60:.1f} min")
    print(f"  Models: {final_path}.zip  |  {os.path.join(best_dir, 'best_model.zip')}")
    print("  Evaluate:  python evaluate_drl.py")
    print("=" * 60)
