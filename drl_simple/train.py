"""
Minimal BC + SAC for HVAC: expert replay prefill + recomputed rewards.

Reuses ../drl EnergyPlus env and train_drl constants so obs/action spaces match.

Run:  python train.py
      DRL_SIMPLE_PROFILE=fast   → shorter smoke run (Windows: set DRL_SIMPLE_PROFILE=fast)

Default ``quality`` profile: longer SAC (300k steps), BC epochs from ../drl/train_drl.py,
critic warm-up, 2 gradient steps per env step, γ=0.995, no reward clipping on expert data.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from imitation.algorithms import bc
from imitation.data.types import Transitions

# --------------------------------------------------------------------------- parent drl on path
_REPO = Path(__file__).resolve().parent.parent
_DRL = _REPO / "drl"
sys.path.insert(0, str(_DRL))

from eplus_sim import EnergyPlusEnv, FrameStackWrapper, get_reward  # noqa: E402
from train_drl import (  # noqa: E402
    BC_EPOCHS,
    BC_LR,
    EXPERT_JSON,
    IDF_FILE,
    SAC_BATCH_SIZE,
    SAC_BUFFER_SIZE,
    WEATHER_FILE,
    _build_config,
    _effective_frame_stack,
    _policy_kwargs,
    _transfer_bc_to_sac,
    linear_schedule,
    load_expert_pairs,
)

# --------------------------------------------------------------------------- local I/O
SIMPLE_MODEL_DIR = Path(__file__).resolve().parent / "models"
TRAIN_OUT = Path(__file__).resolve().parent / "drl_simple_output" / "train"
EXPERT_PATH = Path(EXPERT_JSON) if os.path.isabs(EXPERT_JSON) else _DRL / "expert_data_best_v3.json"

# --------------------------------------------------------------------------- training profile
#   quality (default) — longer SAC, stronger BC (uses BC_EPOCHS from ../drl/train_drl.py),
#                       more critic warm-up, 2 grad steps / update, gentler LR, γ closer to 1.
#   fast              — shorter run for debugging.
# Override:  set DRL_SIMPLE_PROFILE=fast
#
_PROFILE = os.environ.get("DRL_SIMPLE_PROFILE", "quality").strip().lower()

if _PROFILE == "fast":
    BC_EPOCHS_LOCAL = min(BC_EPOCHS, 60)
    TOTAL_SAC_STEPS = 80_000
    ACTOR_FREEZE_STEPS = 2_000
    SAC_LR = 3e-4
    ENT_COEF = 0.05
    GAMMA = 0.99
    TAU = 0.005
    GRADIENT_STEPS = 1
    SAC_TRAIN_FREQ = 2
    LEARNING_STARTS = 256
    BATCH_SIZE = SAC_BATCH_SIZE
    REWARD_CLIP = 50.0
else:
    # quality — longer wall time, better chance to beat ~7.3–7.4k € expert band
    BC_EPOCHS_LOCAL = BC_EPOCHS
    TOTAL_SAC_STEPS = 300_000
    ACTOR_FREEZE_STEPS = 8_000
    SAC_LR = 2e-4
    ENT_COEF = 0.025
    GAMMA = 0.995
    TAU = 0.0075
    GRADIENT_STEPS = 2
    SAC_TRAIN_FREQ = 1
    LEARNING_STARTS = 1_024
    BATCH_SIZE = max(SAC_BATCH_SIZE, 384)
    REWARD_CLIP = 0.0


def _wrap_env(base):
    n = _effective_frame_stack()
    if n > 1:
        return Monitor(FrameStackWrapper(base, n_frames=n))
    return Monitor(base)


def make_wrapped_env(config):
    return _wrap_env(EnergyPlusEnv(config))


def _train_config():
    return _build_config(
        IDF_FILE,
        WEATHER_FILE,
        str(TRAIN_OUT),
        "learn",
    )


def _denormalize_obs_row(
    obs_dict: dict,
    obs_keys: list,
    obs_min: list,
    obs_max: list,
) -> dict:
    phys: dict = {}
    for i, k in enumerate(obs_keys):
        vn = float(obs_dict.get(k, 0.0))
        lo, hi = obs_min[i], obs_max[i]
        if hi > lo:
            phys[k] = (vn + 1.0) / 2.0 * (hi - lo) + lo
        else:
            phys[k] = float(lo)
    return phys


def load_expert_for_replay(json_path: Path, config: dict):
    """Build (obs, next_obs, acts, rewards, dones) for SAC replay buffer."""
    with open(json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    obs_keys = config["observations"]
    act_keys = config["rl_actions"]
    obs_min = config["rl_observation_min"]
    obs_max = config["rl_observation_max"]

    nf = _effective_frame_stack()
    obs_list = []
    next_list = []
    act_list = []
    rew_list = []
    done_list = []

    for row in rows:
        o = row["obs"]
        no = row.get("next_obs")
        if no is None:
            continue
        a = row["action"]
        obs_list.append([float(o[k]) for k in obs_keys])
        next_list.append([float(no[k]) for k in obs_keys])
        act_list.append([float(a[k]) for k in act_keys])
        phys_next = _denormalize_obs_row(no, obs_keys, obs_min, obs_max)
        t_roll = phys_next.get("outdoor_temp_rolling_24h", 22.0)
        r = float(get_reward(phys_next, t_roll))
        if REWARD_CLIP > 0:
            r = float(np.clip(r, -REWARD_CLIP, REWARD_CLIP))
        rew_list.append(r)
        done_list.append(bool(row.get("done", False)))

    obs = np.asarray(obs_list, dtype=np.float32)
    next_obs = np.asarray(next_list, dtype=np.float32)
    acts = np.asarray(act_list, dtype=np.float32)
    rews = np.asarray(rew_list, dtype=np.float32)
    dones = np.asarray(done_list, dtype=np.float32)

    if nf > 1:
        raise RuntimeError("drl_simple does not support frame stacking; set USE_GRU_AND_FRAME_STACK=False in train_drl.py")

    return obs, next_obs, acts, rews, dones


def _prefill_replay_buffer(model: SAC, obs, next_obs, acts, rews, dones) -> None:
    buf = model.replay_buffer
    n = obs.shape[0]
    for i in range(n):
        buf.add(
            obs[i],
            next_obs[i],
            acts[i],
            rews[i],
            dones[i],
            [{}],
        )


if __name__ == "__main__":
    t0 = time.time()
    cfg = _train_config()
    TRAIN_OUT.mkdir(parents=True, exist_ok=True)
    SIMPLE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not EXPERT_PATH.is_file():
        raise FileNotFoundError(f"Expert data not found: {EXPERT_PATH}")

    train_env = make_wrapped_env(cfg)
    pk = _policy_kwargs()

    print("=" * 60)
    print("  drl_simple — BC + SAC (expert replay prefill)")
    print(f"    profile:   {_PROFILE}")
    print(f"    expert:    {EXPERT_PATH}")
    print(f"    BC epochs: {BC_EPOCHS_LOCAL}  (train_drl.BC_EPOCHS={BC_EPOCHS})")
    print(f"    SAC:       {TOTAL_SAC_STEPS:,} steps  freeze_critic={ACTOR_FREEZE_STEPS:,}")
    print(f"             lr={SAC_LR}  ent={ENT_COEF}  γ={GAMMA}  τ={TAU}")
    print(f"             batch={BATCH_SIZE}  grad_steps={GRADIENT_STEPS}  train_freq={SAC_TRAIN_FREQ}")
    print("=" * 60)

    # --- BC (same loader as train_drl: rolling 24h + normalization)
    print("\n  [1/3] Behavioral cloning")
    obs_b, acts_b = load_expert_pairs(str(EXPERT_PATH), cfg)
    expert_data = Transitions(
        obs=obs_b,
        acts=acts_b,
        infos=np.array([{}] * len(obs_b), dtype=object),
        next_obs=np.zeros_like(obs_b),
        dones=np.zeros(len(obs_b), dtype=bool),
    )
    bc_policy = ActorCriticPolicy(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        lr_schedule=lambda _: BC_LR,
        net_arch=pk.get("net_arch", [256, 256]),
        activation_fn=torch.nn.ReLU,
    )
    bc_trainer = bc.BC(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        demonstrations=expert_data,
        rng=np.random.default_rng(42),
        device="auto",
        policy=bc_policy,
        batch_size=BATCH_SIZE,
    )
    bc_trainer.train(n_epochs=BC_EPOCHS_LOCAL)
    bc_path = SIMPLE_MODEL_DIR / "bc_policy.pt"
    bc_trainer.policy.save(str(bc_path))
    print(f"  BC saved  {bc_path}")

    # --- Expert transitions for critic warm-start
    print("\n  [2/3] Load expert transitions + rewards for replay buffer")
    ex_obs, ex_next, ex_act, ex_rew, ex_done = load_expert_for_replay(EXPERT_PATH, cfg)
    print(f"  {ex_obs.shape[0]} transitions  reward mean={ex_rew.mean():.4f} std={ex_rew.std():.4f}")

    # --- SAC
    print("\n  [3/3] SAC (prefilled buffer + fine-tune)")
    sac_model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=linear_schedule(SAC_LR),
        batch_size=BATCH_SIZE,
        buffer_size=SAC_BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        gamma=GAMMA,
        tau=TAU,
        ent_coef=ENT_COEF,
        train_freq=SAC_TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        verbose=1,
        tensorboard_log=str(SIMPLE_MODEL_DIR / "tb_sac"),
        policy_kwargs=pk,
    )
    _transfer_bc_to_sac(bc_trainer.policy, sac_model)
    _prefill_replay_buffer(sac_model, ex_obs, ex_next, ex_act, ex_rew, ex_done)

    for p in sac_model.policy.actor.parameters():
        p.requires_grad = False
    sac_model.learn(total_timesteps=ACTOR_FREEZE_STEPS, reset_num_timesteps=True)

    for p in sac_model.policy.actor.parameters():
        p.requires_grad = True
    sac_model.learn(
        total_timesteps=TOTAL_SAC_STEPS,
        progress_bar=True,
        reset_num_timesteps=False,
    )

    out = SIMPLE_MODEL_DIR / "best_model"
    sac_model.save(str(out))
    print()
    print("=" * 60)
    print(f"  Done in {(time.time() - t0) / 60:.1f} min")
    print(f"  Model: {out}.zip")
    print("  Evaluate:  python evaluate.py")
    print("=" * 60)
