"""
train_transformer.py — BC + SAC with a Transformer encoder over frame-stacked obs.

Reuses the same EnergyPlus env, observation/action specs, and expert data as
``drl/train_drl.py``. Outputs:

  • drl_transformer/models/best_model_sac/best_model.zip
  • drl_transformer/drl_transformer_output/train/run_*/  (EnergyPlus scratch)

Evaluate with ``python evaluate_transformer.py`` for the same CSV column layout
as ``drl/evaluate_drl.py`` (RL log + eplusout.csv under eval/run_*).
"""

import os
import sys

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from imitation.algorithms import bc
from imitation.data.types import Transitions

# ---------------------------------------------------------------------------
# Resolve drl/ (EnergyPlus env, shared config helpers)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
_DRL = os.path.join(_REPO, "drl")
# Load train_drl + eplus_sim from sibling drl/ (do not insert drl before the script dir
# or ``import feature_extractors`` inside train_drl would be ambiguous).
if _DRL not in sys.path:
    sys.path.append(_DRL)

import train_drl as td  # noqa: E402
from eplus_sim import EnergyPlusEnv, FrameStackWrapper  # noqa: E402

from transformer_encoder import TransformerFeatureExtractor  # noqa: E402


# --- Transformer + temporal context ------------------------------------------------
FRAME_STACK_N = 8
D_MODEL = 128
NHEAD = 4
TRANSFORMER_LAYERS = 2
DIM_FEEDFORWARD = 256

BC_EPOCHS = 50
BC_LR = 2e-4

SAC_FROZEN_STEPS = 3_000
SAC_TOTAL_STEPS = 200_000
SAC_BATCH_SIZE = 256
SAC_BUFFER_SIZE = 200_000
SAC_LEARNING_RATE = 3e-4
SAC_TRAIN_FREQ = 2

MODEL_DIR = os.path.join(_HERE, "models")
TRAIN_OUT = os.path.abspath(os.path.join(_HERE, "drl_transformer_output", "train"))

IDF_FILE = os.path.join(_REPO, "DOAS_wNeutralSupplyAir_wFanCoilUnits.idf")
WEATHER_FILE = os.path.join(
    _REPO, "FIN_TR_Tampere.Satakunnankatu.027440_TMYx.2004-2018.epw"
)
EXPERT_JSON = os.path.join(_DRL, "expert_data_best_v3.json")

N_OBS_FEATURES = td.N_OBS_FEATURES


def make_wrapped_env(config):
    """Must match training when loading in evaluate_transformer.py."""
    return Monitor(FrameStackWrapper(EnergyPlusEnv(config), n_frames=FRAME_STACK_N))


def _policy_kwargs():
    return {
        "features_extractor_class": TransformerFeatureExtractor,
        "features_extractor_kwargs": {
            "n_frames": FRAME_STACK_N,
            "n_features_per_frame": N_OBS_FEATURES,
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_layers": TRANSFORMER_LAYERS,
            "dim_feedforward": DIM_FEEDFORWARD,
        },
        "net_arch": [256, 256],
        "share_features_extractor": False,
    }


def _transfer_bc_to_sac(bc_policy, sac_model):
    sac_model.policy.actor.features_extractor.load_state_dict(
        bc_policy.features_extractor.state_dict()
    )
    sac_model.policy.actor.latent_pi.load_state_dict(
        bc_policy.mlp_extractor.policy_net.state_dict()
    )
    sac_model.policy.actor.mu.load_state_dict(bc_policy.action_net.state_dict())
    for net in (sac_model.policy.critic, sac_model.policy.critic_target):
        net.features_extractor.load_state_dict(bc_policy.features_extractor.state_dict())


def linear_schedule(initial_lr: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_lr

    return func


if __name__ == "__main__":
    import time as _time

    _t0 = _time.time()

    train_config = td._build_config(IDF_FILE, WEATHER_FILE, TRAIN_OUT, "learn")
    train_env = make_wrapped_env(train_config)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.join(MODEL_DIR, "sac_final"), exist_ok=True)

    print("=" * 60)
    print("  Transformer + SAC training")
    print(f"    frame_stack: {FRAME_STACK_N}  d_model={D_MODEL}")
    print(f"    SAC frozen: {SAC_FROZEN_STEPS:,}  train: {SAC_TOTAL_STEPS:,}")
    print("=" * 60)

    print()
    print("  STEP 1 / 2 — Behavioral Cloning")

    obs, acts = td.load_expert_pairs(EXPERT_JSON, train_config, n_frames=FRAME_STACK_N)
    acts = np.clip(acts, -0.9999, 0.9999)
    acts = np.arctanh(acts)

    print(f"  Loaded {len(obs)} transitions  obs {obs.shape}  acts {acts.shape}")

    expert_data = Transitions(
        obs=obs,
        acts=acts,
        infos=np.array([{}] * len(obs), dtype=object),
        next_obs=np.zeros_like(obs),
        dones=np.zeros(len(obs), dtype=bool),
    )

    pk = _policy_kwargs()
    bc_policy = ActorCriticPolicy(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        lr_schedule=lambda _: BC_LR,
        net_arch=pk["net_arch"],
        activation_fn=torch.nn.ReLU,
        features_extractor_class=pk["features_extractor_class"],
        features_extractor_kwargs=pk["features_extractor_kwargs"],
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
    print("  Evaluate:  python evaluate_transformer.py")
    print("=" * 60)
