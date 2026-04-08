#!/usr/bin/env python3
"""Run an EnergyPlus IDF simulation with Rule-Based Control (RBC).

This script:
1. Creates an EnergyPlus API instance and simulation state.
2. Registers the RBC controller callbacks (handle init, control logic,
   and optional API-data dump).
3. Launches the EnergyPlus simulation and returns the exit code.

"""

import sys
from pathlib import Path

# --- EnergyPlus Python API setup ---
ENERGYPLUS_DIR = r"C:\EnergyPlusV25-2-0"
sys.path.append(ENERGYPLUS_DIR)
from pyenergyplus.api import EnergyPlusAPI  # noqa: E402

from energyplus_controller import EnergyPlusController
from rbc_model import RBCModel

# --- Paths ---
IDF_FILE = Path(r"../DOAS_wNeutralSupplyAir_wFanCoilUnits.idf")
EPW_FILE = Path(r"../FIN_TR_Tampere.Satakunnankatu.027440_TMYx.2004-2018.epw")
OUT_DIR = Path(__file__).resolve().parent / "eplus_out"


def run_with_api(idf: Path, epw: Path | None, outdir: Path) -> int:
    """Run a single EnergyPlus simulation with RBC control.

    Args:
        idf:    Path to the ``.idf`` building-model file.
        epw:    Path to the ``.epw`` weather file (``None`` for design-day only).
        outdir: Directory for EnergyPlus output files.

    Returns:
        EnergyPlus return code (0 = success).
    """
    api = EnergyPlusAPI()
    state = api.state_manager.new_state()

    # --- Instantiate control strategy ---
    model = RBCModel()
    controller = EnergyPlusController(api, model)

    # --- Register runtime callbacks ---
    # 1. Initialise sensor/actuator handles once warm-up is done
    api.runtime.callback_after_new_environment_warmup_complete(
        state, controller.initialize_handles
    )
    # 2. Apply control logic at the start of every zone timestep
    api.runtime.callback_begin_zone_timestep_after_init_heat_balance(
        state, controller.control_callback
    )
    # 3. (Optional) Dump all available API data after system sizing
    api.runtime.callback_end_system_sizing(
        state, controller.get_api_data
    )

    # --- Build command-line arguments for EnergyPlus ---
    args: list[str] = ["-d", str(outdir)]
    if epw is not None:
        args += ["-w", str(epw), "-r"]
    args.append(str(idf))

    print("Starting EnergyPlus with args:", args)

    try:
        rc = api.runtime.run_energyplus(state, args)
    except TypeError:
        # Fallback for older pyenergyplus versions
        rc = api.runtime.run_energyplus(args)

    # Mark final transition as terminal if any data was collected
    if controller.trajectories:
        controller.trajectories[-1]["done"] = True

    # Save expert data
    try:
        TRAJ_JSON = OUT_DIR / "expert_data.json"
        controller.save_trajectories(str(TRAJ_JSON))
        print(f"Saved expert trajectories to: {TRAJ_JSON}")
    except Exception as e:
        print(f"WARNING: failed to save JSON trajectories: {e}")

    return int(rc)


def main() -> None:
    """Run the simulation with the provided paths."""
    if not IDF_FILE.exists():
        sys.exit(f"ERROR: IDF file not found: {IDF_FILE}")
    if not EPW_FILE.exists():
        sys.exit(f"ERROR: Weather file not found: {EPW_FILE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rc = run_with_api(IDF_FILE, EPW_FILE, OUT_DIR)

    if rc == 0:
        print(f"Simulation completed successfully. Output in: {OUT_DIR.resolve()}")
    else:
        print(f"Simulation FAILED (return code {rc}).")

    sys.exit(rc if rc is not None else 1)


if __name__ == "__main__":
    main()
