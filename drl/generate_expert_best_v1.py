#!/usr/bin/env python3
"""Run EnergyPlus with rbc_scheduled_research/best_v1_7482eur RBC and write
expert trajectories for behavioral cloning in train_drl.py.

Normalization uses OBS_SPEC / ACTION_SPEC from train_drl.py so the JSON matches
EnergyPlusEnv observation/action spaces.

The exported observations now include:
    - Zone occupancy (space{i}_occ)
    - Cyclical time features (hour_sin/cos, day_sin/cos, month_sin/cos)
    - 24 h rolling outdoor temperature
    - Raw time metadata (_raw_hour, _raw_day, _raw_month, _raw_outdoor_temp)
      so train_drl.load_expert_pairs can recompute features if needed.

Usage:
    cd drl
    python generate_expert_best_v1.py

Output:
    drl/expert_data_best_v1.json
    drl/eplus_out_expert_best_v1/
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

ENERGYPLUS_DIR = r"C:\EnergyPlusV25-2-0"
sys.path.append(ENERGYPLUS_DIR)
from pyenergyplus.api import EnergyPlusAPI  # noqa: E402

_DRL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DRL_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "rbc_scheduled_research"))

from best_v1_7482eur import (  # noqa: E402
    ACTUATORS,
    METERS,
    PARAMS,
    RBCModel,
    VARIABLES,
)

sys.path.insert(0, str(_DRL_DIR))
from train_drl import ACTION_SPEC, OBS_SPEC  # noqa: E402

IDF_FILE = _REPO_ROOT / "DOAS_wNeutralSupplyAir_wFanCoilUnits.idf"
EPW_FILE = _REPO_ROOT / "FIN_TR_Tampere.Satakunnankatu.027440_TMYx.2004-2018.epw"
EPLUS_OUT_DIR = _DRL_DIR / "eplus_out_expert_best_v1"
EXPERT_JSON_PATH = _DRL_DIR / "expert_data_best_v1.json"


def _normalize_obs(obs: Dict[str, float]) -> Dict[str, float]:
    """Normalise raw observation dict to [-1, 1] using OBS_SPEC bounds."""
    out: Dict[str, float] = {}
    for key, (lo, hi) in OBS_SPEC.items():
        v = float(obs.get(key, 0.0))
        v = max(lo, min(hi, v))
        if hi > lo:
            out[key] = 2.0 * (v - lo) / (hi - lo) - 1.0
        else:
            out[key] = 0.0
    return out


def _normalize_action(action: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, (lo, hi) in ACTION_SPEC.items():
        v = float(action[key])
        v = max(lo, min(hi, v))
        if hi > lo:
            out[key] = 2.0 * (v - lo) / (hi - lo) - 1.0
        else:
            out[key] = 0.0
    return out


def _collect_obs(api_obj: Any, handles: Dict[str, int], state: Any,
                 outdoor_temp_rolling_24h: float) -> Dict[str, float]:
    """Collect raw sensor values and compute all derived features.

    Returns a dict whose keys cover the full OBS_SPEC plus ``_raw_*``
    metadata fields used by ``train_drl.load_expert_pairs``.
    """
    def get_var(name: str, default: float = 0.0) -> float:
        h = handles.get(name)
        if h is None or h == -1:
            return default
        if name in METERS:
            return float(api_obj.exchange.get_meter_value(state, h))
        return float(api_obj.exchange.get_variable_value(state, h))

    hour  = float(api_obj.exchange.hour(state))
    day   = float(api_obj.exchange.day_of_week(state))
    month = float(api_obj.exchange.month(state))

    obs: Dict[str, float] = {
        "outdoor_temp": get_var("outdoor_temp"),
        "plenum_temp":  get_var("plenum_temp"),

        "space1_temp": get_var("space1_temp"),
        "space1_rh":   get_var("space1_rh"),
        "space1_co2":  get_var("space1_co2"),
        "space1_occ":  get_var("space1_occupancy", 0.0),

        "space2_temp": get_var("space2_temp"),
        "space2_rh":   get_var("space2_rh"),
        "space2_co2":  get_var("space2_co2"),
        "space2_occ":  get_var("space2_occupancy", 0.0),

        "space3_temp": get_var("space3_temp"),
        "space3_rh":   get_var("space3_rh"),
        "space3_co2":  get_var("space3_co2"),
        "space3_occ":  get_var("space3_occupancy", 0.0),

        "space4_temp": get_var("space4_temp"),
        "space4_rh":   get_var("space4_rh"),
        "space4_co2":  get_var("space4_co2"),
        "space4_occ":  get_var("space4_occupancy", 0.0),

        "space5_temp": get_var("space5_temp"),
        "space5_rh":   get_var("space5_rh"),
        "space5_co2":  get_var("space5_co2"),
        "space5_occ":  get_var("space5_occupancy", 0.0),

        "electricity_hvac": get_var("electricity_hvac"),
        "gas_total":        get_var("gas_total"),

        # Cyclical time encoding
        "hour_sin":  math.sin(2.0 * math.pi * hour / 24.0),
        "hour_cos":  math.cos(2.0 * math.pi * hour / 24.0),
        "day_sin":   math.sin(2.0 * math.pi * (day - 1) / 7.0),
        "day_cos":   math.cos(2.0 * math.pi * (day - 1) / 7.0),
        "month_sin": math.sin(2.0 * math.pi * (month - 1) / 12.0),
        "month_cos": math.cos(2.0 * math.pi * (month - 1) / 12.0),

        # Rolling outdoor temperature (caller maintains the buffer)
        "outdoor_temp_rolling_24h": outdoor_temp_rolling_24h,

        # Raw metadata for downstream recomputation
        "_raw_hour":         hour,
        "_raw_day":          day,
        "_raw_month":        month,
        "_raw_outdoor_temp": get_var("outdoor_temp"),
    }
    return obs


class ExpertTrajectoryController:
    def __init__(self, api_obj: Any, model: RBCModel) -> None:
        self.api = api_obj
        self.model = model
        self.handles: Dict[str, int] = {}
        self.prev_obs: Optional[Dict[str, float]] = None
        self.prev_action: Optional[Dict[str, float]] = None
        self.trajectories: List[Dict[str, Any]] = []
        self._outdoor_temp_buf: deque = deque(maxlen=96)

    def initialize_handles(self, state: Any) -> None:
        ex = self.api.exchange
        for name, (var, key) in VARIABLES.items():
            self.handles[name] = ex.get_variable_handle(state, var, key)
        for name, (ctype, control, key) in ACTUATORS.items():
            self.handles[name] = ex.get_actuator_handle(state, ctype, control, key)
        for name, meter_name in METERS.items():
            self.handles[name] = ex.get_meter_handle(state, meter_name)
        bad = [n for n, h in self.handles.items() if h == -1]
        if bad:
            print(f"WARNING: unresolved handles: {bad}")
        else:
            print(f"All {len(self.handles)} handles OK.")

    def set_actuator(self, name: str, value: float, state: Any) -> None:
        handle = self.handles.get(name)
        if handle is not None and handle != -1:
            self.api.exchange.set_actuator_value(state, handle, value)

    def get_variable(self, name: str, state: Any, default: float = 0.0) -> float:
        handle = self.handles.get(name)
        if handle is None or handle == -1:
            return default
        if name in METERS:
            return float(self.api.exchange.get_meter_value(state, handle))
        return float(self.api.exchange.get_variable_value(state, handle))

    def _outdoor_rolling(self) -> float:
        if self._outdoor_temp_buf:
            return sum(self._outdoor_temp_buf) / len(self._outdoor_temp_buf)
        return 0.0

    def control_callback(self, state: Any) -> None:
        if not self.handles:
            return

        # Update rolling outdoor temperature buffer
        ot = self.get_variable("outdoor_temp", state)
        self._outdoor_temp_buf.append(ot)

        raw_obs = _collect_obs(self.api, self.handles, state,
                               self._outdoor_rolling())

        outdoor_temp = raw_obs["outdoor_temp"]
        plenum_temp = raw_obs["plenum_temp"]
        direct_solar = self.get_variable("direct_solar", state)
        wind_speed = self.get_variable("wind_speed", state)
        hour = raw_obs["_raw_hour"]
        day = raw_obs["_raw_day"]

        temps, co2s, occs = [], [], []
        for i in range(1, 6):
            temps.append(self.get_variable(f"space{i}_temp", state))
            co2s.append(self.get_variable(f"space{i}_co2", state))
            occs.append(self.get_variable(f"space{i}_occupancy", state))

        avg_temp = sum(temps) / len(temps)
        max_co2 = max(co2s)
        total_occupancy = sum(occs)

        htg, clg, supply_air_temp, flow = self.model.calculate_setpoints(
            zone_temp=avg_temp,
            outdoor_temp=outdoor_temp,
            return_air_temp=plenum_temp,
            occupancy=total_occupancy,
            hour=hour,
            day=day,
            co2_concentration=max_co2,
            direct_solar=direct_solar,
            wind_speed=wind_speed,
        )

        self.set_actuator("htg_setpoint", htg, state)
        self.set_actuator("clg_setpoint", clg, state)
        self.set_actuator("ahu_temperature_setpoint", supply_air_temp, state)
        self.set_actuator("ahu_mass_flow_rate_setpoint", flow, state)

        action_phys = {
            "cooling_setpoint": clg,
            "heating_setpoint": htg,
            "ahu_supply_temp": supply_air_temp,
            "supply_fan_flow": flow,
        }

        normalized_obs = _normalize_obs(raw_obs)
        normalized_action = _normalize_action(action_phys)

        # Preserve raw metadata alongside the normalised observation
        normalized_obs["_raw_hour"]         = raw_obs["_raw_hour"]
        normalized_obs["_raw_day"]          = raw_obs["_raw_day"]
        normalized_obs["_raw_month"]        = raw_obs["_raw_month"]
        normalized_obs["_raw_outdoor_temp"] = raw_obs["_raw_outdoor_temp"]

        if self.prev_obs is not None:
            self.trajectories.append({
                "obs": self.prev_obs,
                "action": self.prev_action,
                "reward": 0.0,
                "next_obs": normalized_obs,
                "done": False,
            })

        self.prev_obs = normalized_obs
        self.prev_action = normalized_action


def run_export(params: Dict[str, Any] | None = None) -> int:
    params = PARAMS if params is None else params

    if not IDF_FILE.is_file():
        print(f"ERROR: IDF not found: {IDF_FILE}")
        return 1
    if not EPW_FILE.is_file():
        print(f"ERROR: Weather not found: {EPW_FILE}")
        return 1

    if EPLUS_OUT_DIR.exists():
        shutil.rmtree(EPLUS_OUT_DIR)
    EPLUS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    api_obj = EnergyPlusAPI()
    ep_state = api_obj.state_manager.new_state()

    for _name, (var, key) in VARIABLES.items():
        api_obj.exchange.request_variable(ep_state, var, key)

    model = RBCModel(params)
    controller = ExpertTrajectoryController(api_obj, model)

    api_obj.runtime.callback_after_new_environment_warmup_complete(
        ep_state, controller.initialize_handles
    )
    api_obj.runtime.callback_begin_zone_timestep_after_init_heat_balance(
        ep_state, controller.control_callback
    )

    args = ["-d", str(EPLUS_OUT_DIR), "-w", str(EPW_FILE), "-r", str(IDF_FILE)]
    print("Running EnergyPlus for expert export:", args)
    t0 = time.time()
    rc = api_obj.runtime.run_energyplus(ep_state, args)
    print(f"EnergyPlus finished in {time.time() - t0:.1f}s, code {rc}")

    if controller.trajectories:
        controller.trajectories[-1]["done"] = True

    with open(EXPERT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(controller.trajectories, f)

    print(f"Wrote {len(controller.trajectories)} transitions to {EXPERT_JSON_PATH}")
    return int(rc)


if __name__ == "__main__":
    sys.exit(run_export())
