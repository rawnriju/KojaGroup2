drl_simple — minimal BC + SAC pipeline with expert replay prefill
===================================================================

Why this exists
---------------
The original drl/train_drl.py feeds BC from expert data, but SAC starts with an
empty replay buffer and expert transitions had reward=0 in the JSON. The critic
then learns only from sparse online samples, so the policy often drifts without
beating the RBC expert.

This folder reuses the EnergyPlus env from ../drl/ but:
  • Recomputes per-step rewards from expert next_obs (same formula as eplus_sim).
  • Prefills SAC’s replay buffer with those transitions before fine-tuning.
  • Uses lower entropy / shorter “frozen actor” so exploration stays near the expert.

Run (from this directory, same Python env you use for ../drl)
--------------------------------------------------------------
  cd drl_simple
  python train.py

Evaluate (full-year sim, writes drl_simple_output/eval/...)
----------------------------------------------------------
  python evaluate.py

Inputs
------
  ../drl/expert_data_best_v3.json   (from generate_expert_best_v1.py)

Outputs
-------
  models/bc_policy.pt
  models/best_model.zip
  drl_simple_output/train/   — EnergyPlus scratch during SAC
  drl_simple_output/eval/    — use eplusout.csv for notebooks

Training profiles (train.py)
----------------------------
  Default is **quality** (longer, tuned for lower total cost vs ~7.3k € expert):
    • BC epochs = BC_EPOCHS in ../drl/train_drl.py (e.g. 500)
    • SAC 300 000 env steps, 8 000 steps with actor frozen (critic-only warm-up)
    • Learning rate 2e-4 (linear decay), γ=0.995, τ=0.0075, batch ≥384
    • 2 gradient steps per update, train_freq=1, expert replay rewards **unclipped**

  **fast** (debug / shorter run):
        Windows CMD:  set DRL_SIMPLE_PROFILE=fast && python train.py
        PowerShell:   $env:DRL_SIMPLE_PROFILE="fast"; python train.py

  To push even longer, edit train.py (quality branch): raise TOTAL_SAC_STEPS
  (e.g. 350k–400k) or ACTOR_FREEZE_STEPS (e.g. 12k).
