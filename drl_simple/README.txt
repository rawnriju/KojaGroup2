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

You can tune TOTAL_SAC_STEPS, ENT_COEF, and BC_EPOCHS at the top of train.py.
