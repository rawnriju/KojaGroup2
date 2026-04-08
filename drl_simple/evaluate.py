"""Run one full-year evaluation with drl_simple trained policy."""

import sys
from pathlib import Path

import pandas as pd
from stable_baselines3 import SAC

_REPO = Path(__file__).resolve().parent.parent
_DRL = _REPO / "drl"
sys.path.insert(0, str(_DRL))

from train_drl import (  # noqa: E402
    IDF_FILE,
    N_OBS_FEATURES,
    WEATHER_FILE,
    _build_config,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import make_wrapped_env  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent / "models" / "best_model.zip"
EVAL_OUT = Path(__file__).resolve().parent / "drl_simple_output" / "eval"


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Train first — missing {MODEL_PATH}")

    cfg = _build_config(
        IDF_FILE,
        WEATHER_FILE,
        str(EVAL_OUT),
        "test",
    )
    EVAL_OUT.mkdir(parents=True, exist_ok=True)

    env = make_wrapped_env(cfg)
    model = SAC.load(str(MODEL_PATH), env=env)

    obs_names = cfg["observations"]
    act_names = cfg["rl_actions"]

    obs, _ = env.reset()
    done, truncated = False, False
    rows = []

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, truncated, _ = env.step(action)
        row = {name: float(obs[i]) for i, name in enumerate(obs_names)}
        row.update({name: float(action[i]) for i, name in enumerate(act_names)})
        row["reward"] = float(reward)
        rows.append(row)
        obs = next_obs

    env.close()

    df = pd.DataFrame(rows)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    log_csv = Path(__file__).resolve().parent / f"sac_simple_eval_{ts}.csv"
    df.to_csv(log_csv, index=False)

    run_dirs = sorted(
        d for d in EVAL_OUT.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    )
    latest_csv = run_dirs[-1] / "eplusout.csv" if run_dirs else None

    print()
    print("=" * 60)
    print(f"  drl_simple evaluation — {len(df)} steps  obs_dim={N_OBS_FEATURES}")
    print(f"  RL log: {log_csv}")
    if latest_csv and latest_csv.is_file():
        print(f"  EnergyPlus: {latest_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
