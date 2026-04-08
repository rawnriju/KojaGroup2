"""
eplus_sim.py — Gymnasium environment coupling EnergyPlus to an RL agent.

Architecture
------------
EnergyPlus runs in a daemon thread; the RL agent runs in the main thread.
Synchronisation is achieved via three ``threading.Event`` objects:

* **obs_event** - E+ callback signals that new observations are available.
* **act_event** - RL agent signals that the next action has been chosen.
* **stop_event** - RL agent requests graceful shutdown of the E+ thread.

Each simulation timestep follows: read sensors → notify RL → wait for action
→ apply actuators → advance E+.

Key external dependencies:
    - ``pyenergyplus``  (EnergyPlus Python API)
    - ``gymnasium``     (RL environment interface)
    - ``variable_config`` (sensor / meter / actuator definitions for this model)
"""

import sys
import os
import math
import datetime
import time
import threading
from collections import deque
from typing import Dict, Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

ENERGYPLUS_DIR = r"C:\EnergyPlusV25-2-0"
sys.path.append(ENERGYPLUS_DIR)

from pyenergyplus.api import EnergyPlusAPI

from logger_eplus import logger

from variable_config import SENSOR_DEF, METER_DEF, ACTUATOR_DEF


# ---------------------------------------------------------------------------
# Global state shared between the E+ callback thread and the RL main thread
# ---------------------------------------------------------------------------

api = EnergyPlusAPI()

act_event  = threading.Event()
obs_event  = threading.Event()
stop_event = threading.Event()

eplus_data_collection = []
actions_list = []
eplus_sim_step = 0

_eplus_running = False
_eplus_failed  = False
_eplus_thread  = None
_ep_state      = None
_run_counter   = 0

_handles_initialized = False
_sensor_handles: Dict[str, int] = {}
_actuator_handles: Dict[str, int] = {}
_meter_handles: Dict[str, int] = {}

SAFE_INITIAL_ACTION = {
    "cooling_setpoint": 21.5,
    "heating_setpoint": 21.5,
    "ahu_supply_temp": 19.0,
    "supply_fan_flow": 0.20,
}

POST_WARMUP_HOLD_STEPS = 8
_post_warmup_steps = 0
_warmup_complete_seen = False


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_get(dic: dict, key: str, default=0.0):
    return dic[key] if key in dic else default


# ---------------------------------------------------------------------------
# Finnish S-class temperature bands (from SFS 5511 / EN 16798)
# ---------------------------------------------------------------------------

def compute_s_class_bands(t_out_24h: float):
    """Return (lower_S1, upper_S1, lower_S2, upper_S2, lower_S3, upper_S3).

    ``t_out_24h`` is the 24-hour exponentially-weighted running mean of
    outdoor dry-bulb temperature.  Piecewise-linear formulas match the
    hackathon evaluation notebook.
    """
    t = t_out_24h

    lower_S1 = 20.5 if t <= 0 else (20.5 + 0.075 * t if t <= 20 else 22.0)
    upper_S1 = 22.0 if t <= 0 else (22.5 + 0.166 * t if t <= 15 else 25.0)

    lower_S2 = 20.5 if t <= 0 else (20.5 + 0.025 * t if t <= 20 else 21.0)
    upper_S2 = 23.0 if t <= 0 else (23.0 + 0.200 * t if t <= 15 else 26.0)

    lower_S3 = 20.0
    upper_S3 = 25.0 if t <= 10 else 27.0

    return lower_S1, upper_S1, lower_S2, upper_S2, lower_S3, upper_S3


# ---------------------------------------------------------------------------
# Derived observation features
# ---------------------------------------------------------------------------

def augment_step_data(step_data: Dict[str, Any],
                      outdoor_temp_rolling_24h: float) -> Dict[str, Any]:
    """Add cyclical time and rolling-mean features to raw E+ step data.

    These keys must be present in the step_data already:
    ``hour``, ``day_of_week``, ``month``.
    The result dict is mutated in-place and returned for convenience.
    """
    hour  = _safe_get(step_data, "hour", 0)
    day   = _safe_get(step_data, "day_of_week", 1)
    month = _safe_get(step_data, "month", 1)

    step_data["hour_sin"]  = math.sin(2.0 * math.pi * hour / 24.0)
    step_data["hour_cos"]  = math.cos(2.0 * math.pi * hour / 24.0)
    step_data["day_sin"]   = math.sin(2.0 * math.pi * (day - 1) / 7.0)
    step_data["day_cos"]   = math.cos(2.0 * math.pi * (day - 1) / 7.0)
    step_data["month_sin"] = math.sin(2.0 * math.pi * (month - 1) / 12.0)
    step_data["month_cos"] = math.cos(2.0 * math.pi * (month - 1) / 12.0)

    step_data["outdoor_temp_rolling_24h"] = outdoor_temp_rolling_24h
    return step_data


# ---------------------------------------------------------------------------
# E+ data-exchange helpers (called from the callback thread)
# ---------------------------------------------------------------------------

def _init_handles(state) -> bool:
    global _handles_initialized, _sensor_handles, _actuator_handles, _meter_handles

    if _handles_initialized:
        return True

    exch = api.exchange

    if not exch.api_data_fully_ready(state):
        return False

    _meter_handles = {}
    _sensor_handles = {}
    _actuator_handles = {}

    for alias, var_name, key_value, _ in SENSOR_DEF:
        _sensor_handles[alias] = exch.get_variable_handle(state, var_name, key_value)

    for alias, meter_name in METER_DEF:
        _meter_handles[alias] = exch.get_meter_handle(state, meter_name)

    for alias, comp_type, control_type, key_value in ACTUATOR_DEF:
        _actuator_handles[alias] = exch.get_actuator_handle(state, comp_type, control_type, key_value)

    bad_sensors = [k for k, v in _sensor_handles.items() if v == -1]
    bad_meters = [k for k, v in _meter_handles.items() if v == -1]
    bad_acts = [k for k, v in _actuator_handles.items() if v == -1]

    if bad_sensors:
        logger.warning("Some variable handles were not found: %s", bad_sensors)
    if bad_meters:
        logger.warning("Some meter handles were not found: %s", bad_meters)
    if bad_acts:
        logger.warning("Some actuator handles were not found: %s", bad_acts)

    _handles_initialized = True
    logger.info("EnergyPlus handles initialized")
    return True


def _read_current_timestep(state) -> Dict[str, Any]:
    """Collect all sensor and meter values for the current E+ timestep."""
    exch = api.exchange

    data = {
        "year": exch.year(state),
        "month": exch.month(state),
        "day": exch.day_of_month(state),
        "hour": exch.hour(state),
        "minutes": exch.minutes(state),
        "day_of_week": exch.day_of_week(state),
    }

    for key, handle in _sensor_handles.items():
        if key in {"year", "month", "day", "hour", "minutes", "day_of_week"}:
            continue
        if handle != -1:
            data[key] = exch.get_variable_value(state, handle)
        else:
            data[key] = 0.0

    for key, handle in _meter_handles.items():
        if handle != -1:
            data[key] = exch.get_meter_value(state, handle)
        else:
            data[key] = 0.0

    return data


def _apply_action(state, action_dict: Dict[str, float]):
    exch = api.exchange
    for name, value in action_dict.items():
        handle = _actuator_handles.get(name, -1)
        if handle != -1:
            exch.set_actuator_value(state, handle, float(value))
        else:
            logger.warning("No actuator handle found for action '%s'", name)


# ---------------------------------------------------------------------------
# E+ runtime callback
# ---------------------------------------------------------------------------

def callback_function_bp(state) -> None:
    global eplus_sim_step, _post_warmup_steps, _warmup_complete_seen

    try:
        if not _init_handles(state):
            return

        curr = _read_current_timestep(state)
        eplus_data_collection.append(curr)
        eplus_sim_step += 1

        obs_event.set()

        while not act_event.is_set():
            if stop_event.is_set():
                return
            act_event.wait(timeout=0.1)
        act_event.clear()

        in_warmup = api.exchange.warmup_flag(state)

        if in_warmup:
            action_to_apply = SAFE_INITIAL_ACTION
        else:
            if not _warmup_complete_seen:
                _warmup_complete_seen = True
                _post_warmup_steps = 0

            if _post_warmup_steps < POST_WARMUP_HOLD_STEPS:
                action_to_apply = SAFE_INITIAL_ACTION
                _post_warmup_steps += 1
            elif len(actions_list) > 0 and actions_list[-1] is not None:
                action_to_apply = actions_list[-1]
            else:
                action_to_apply = SAFE_INITIAL_ACTION

        _apply_action(state, action_to_apply)

    except Exception as exc:
        logger.exception("Error in EnergyPlus callback: %s", exc)
        obs_event.set()


# ---------------------------------------------------------------------------
# E+ simulation launcher (runs inside daemon thread)
# ---------------------------------------------------------------------------

def run_energyplus(idf_file: str, weather_file: str, output_path: str, callback) -> int:
    global _ep_state, _handles_initialized, eplus_sim_step, _eplus_running, _eplus_failed, _run_counter

    logger.info("Starting EnergyPlus simulation")
    logger.info("IDF: %s", idf_file)
    logger.info("Weather: %s", weather_file)
    logger.info("Output path: %s", output_path)

    if not os.path.isfile(weather_file):
        logger.error("Weather file does not exist: %s", weather_file)
        _eplus_failed = True
        _eplus_running = False
        obs_event.set()
        return 1

    _handles_initialized = False
    eplus_sim_step = 0
    _eplus_running = True
    _eplus_failed = False

    _run_counter += 1
    run_output_path = os.path.join(output_path, f"run_{_run_counter}")
    os.makedirs(run_output_path, exist_ok=True)

    state = api.state_manager.new_state()
    _ep_state = state

    api.runtime.callback_begin_zone_timestep_after_init_heat_balance(state, callback)

    args = ["-w", weather_file, "-d", run_output_path, idf_file, "-r"]

    result = api.runtime.run_energyplus(state, args)
    logger.info("EnergyPlus finished with exit code %s", result)

    if result != 0:
        _eplus_failed = True
        logger.error("EnergyPlus exited with error code %s", result)

    _eplus_running = False
    obs_event.set()

    try:
        api.state_manager.delete_state(state)
    except Exception:
        pass
    _ep_state = None

    return result


# ---------------------------------------------------------------------------
# Observation / reward helpers
# ---------------------------------------------------------------------------

def get_time(step_data: Dict[str, Any]) -> datetime.datetime:
    year = int(_safe_get(step_data, "year", 2001))
    month = int(_safe_get(step_data, "month", 1))
    day = int(_safe_get(step_data, "day", 1))
    hour = int(_safe_get(step_data, "hour", 0))
    minute = int(_safe_get(step_data, "minutes", 0))

    if minute >= 60:
        minute = 0
        hour += 1
    if hour >= 24:
        hour = 0

    return datetime.datetime(year, month, day, hour, minute)


def get_observations(step_data: Dict[str, Any], config) -> Dict[str, float]:
    """Extract RL observation vector from (augmented) E+ timestep data.

    ``step_data`` should already contain derived keys added by
    ``augment_step_data`` (cyclical time, rolling outdoor temp, etc.).
    """
    obs_keys = config["observations"]
    obs = {k: float(_safe_get(step_data, k, 0.0)) for k in obs_keys}

    if config.get('observation_normalize') == 1:
        obs_min = config['rl_observation_min']
        obs_max = config['rl_observation_max']
        for i, key in enumerate(obs_keys):
            lo, hi = obs_min[i], obs_max[i]
            if hi > lo:
                obs[key] = float(2.0 * (np.clip(obs[key], lo, hi) - lo) / (hi - lo) - 1.0)
            else:
                obs[key] = 0.0

    return obs


def get_reward(step_data: Dict[str, Any],
               outdoor_temp_rolling_24h: float = 22.0) -> float:
    """Compute the scalar reward for the current timestep.

    Improvements over the original:
    1. Proper Finnish S1/S2/S3 temperature bands based on 24 h running mean
       of outdoor temperature (matching hackathon evaluation notebooks).
    2. Smooth reward shaping: continuous penalty that increases near
       thresholds, giving the policy useful gradients everywhere.
    3. CO2 proximity penalty: quadratic cost that rises before discrete
       thresholds, encouraging proactive ventilation.

    Discrete penalty rates (€/h) match the hackathon scoring:
        CO2:  >770 → 2, >970 → 10, >1220 → 50
        Temp: outside S1 → 1, outside S2 → 5, outside S3 → 25
    """
    TIMESTEP_HOURS = 0.25

    # ── Energy cost ──────────────────────────────────────────────────────
    elec_j = _safe_get(step_data, "electricity_hvac", 0.0)
    gas_j  = _safe_get(step_data, "gas_total", 0.0)
    elec_kwh = elec_j / 3_600_000.0
    gas_kwh  = gas_j  / 3_600_000.0
    energy_cost = elec_kwh * 0.11 + gas_kwh * 0.06

    # ── CO2 penalty (per zone) ───────────────────────────────────────────
    co2_cost = 0.0
    for i in range(1, 6):
        co2 = _safe_get(step_data, f"space{i}_co2", 400.0)

        if co2 > 1220:
            co2_cost += 50.0 * TIMESTEP_HOURS
        elif co2 > 970:
            co2_cost += 10.0 * TIMESTEP_HOURS
        elif co2 > 770:
            co2_cost += 2.0 * TIMESTEP_HOURS

        # Smooth shaping: quadratic cost that rises as CO2 approaches 770 ppm
        if co2 > 550:
            proximity = max(0.0, (co2 - 550.0) / 670.0)  # 0→1 over 550→1220
            co2_cost += 0.3 * TIMESTEP_HOURS * proximity ** 2

    # ── Temperature penalty (per zone, proper S-class bands) ─────────────
    bands = compute_s_class_bands(outdoor_temp_rolling_24h)
    lower_S1, upper_S1, lower_S2, upper_S2, lower_S3, upper_S3 = bands

    temp_cost = 0.0
    for i in range(1, 6):
        temp = _safe_get(step_data, f"space{i}_temp", 22.0)

        in_s1 = lower_S1 <= temp <= upper_S1
        in_s2 = lower_S2 <= temp <= upper_S2
        in_s3 = lower_S3 <= temp <= upper_S3

        if not in_s3:
            temp_cost += 25.0 * TIMESTEP_HOURS
        elif not in_s2:
            temp_cost += 5.0 * TIMESTEP_HOURS
        elif not in_s1:
            temp_cost += 1.0 * TIMESTEP_HOURS

        # Smooth shaping: penalise distance from S1 centre, so the policy
        # prefers the middle of the comfort band rather than just any point
        # inside S1.
        center = (lower_S1 + upper_S1) / 2.0
        half_width = max((upper_S1 - lower_S1) / 2.0, 0.5)
        normalised_dev = abs(temp - center) / half_width
        if normalised_dev > 0.5:
            temp_cost += 0.15 * TIMESTEP_HOURS * (normalised_dev - 0.5) ** 2

    total_cost = energy_cost + co2_cost + temp_cost
    return -total_cost


# ---------------------------------------------------------------------------
# Frame-stack Gymnasium wrapper
# ---------------------------------------------------------------------------

class FrameStackWrapper(gym.Wrapper):
    """Stack the last *n_frames* observations into a single flat vector.

    Works transparently with off-policy replay buffers because the full
    stacked vector is stored as the observation.
    """

    def __init__(self, env: gym.Env, n_frames: int = 4):
        super().__init__(env)
        self.n_frames = n_frames
        self._obs_size = env.observation_space.shape[0]
        lo = np.tile(env.observation_space.low,  n_frames)
        hi = np.tile(env.observation_space.high, n_frames)
        self.observation_space = Box(lo, hi, dtype=np.float32)
        self._frames: deque = deque(maxlen=n_frames)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self._frames.append(obs.copy())
        return np.concatenate(list(self._frames)), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self._frames.append(obs.copy())
        return np.concatenate(list(self._frames)), reward, done, truncated, info


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------

class EnergyPlusEnv(gym.Env):
    """Gymnasium-compatible wrapper around EnergyPlus.

    E+ runs in a daemon thread; this class exposes the standard
    ``reset()`` / ``step(action)`` / ``close()`` interface expected
    by RL training loops (e.g. Stable-Baselines3).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.get('actuator_normalize') == 1:
            self.action_space = Box(-1, 1,
                                    shape=(len(config['rl_actions']),),
                                    dtype=np.float32)
        else:
            self.action_space = Box(
                np.array(config['rl_action_min'], dtype=np.float32),
                np.array(config['rl_action_max'], dtype=np.float32),
                dtype=np.float32)
        logger.info('action_space: %s', self.action_space)

        if config.get('observation_normalize') == 1:
            self.observation_space = Box(-1, 1,
                                         shape=(len(config['observations']),),
                                         dtype=np.float32)
        else:
            self.observation_space = Box(
                np.array(config['rl_observation_min'], dtype=np.float32),
                np.array(config['rl_observation_max'], dtype=np.float32),
                dtype=np.float32)
        logger.info('observation_space: %s', self.observation_space)

        self.rl_data_collection = []
        self._outdoor_temp_history: deque = deque(maxlen=96)  # 24 h @ 15 min

    # ----- internal helpers ------------------------------------------------

    def _get_outdoor_rolling(self) -> float:
        if self._outdoor_temp_history:
            return sum(self._outdoor_temp_history) / len(self._outdoor_temp_history)
        return 0.0

    def _prepare_step_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Update rolling buffer, augment with derived features."""
        outdoor_temp = _safe_get(raw, "outdoor_temp", 0.0)
        self._outdoor_temp_history.append(outdoor_temp)
        return augment_step_data(raw, self._get_outdoor_rolling())

    def _start_energyplus(self):
        global _eplus_thread, _eplus_failed
        global _post_warmup_steps, _warmup_complete_seen
        _post_warmup_steps = 0
        _warmup_complete_seen = False

        stop_event.set()
        act_event.set()
        if _eplus_thread is not None and _eplus_thread.is_alive():
            logger.info("Joining previous EnergyPlus thread …")
            _eplus_thread.join(timeout=30)

        eplus_data_collection.clear()
        actions_list.clear()
        obs_event.clear()
        act_event.clear()
        stop_event.clear()
        _eplus_failed = False
        self._outdoor_temp_history.clear()

        logger.info('Starting new EnergyPlus run')
        _eplus_thread = threading.Thread(
            target=run_energyplus,
            args=(self.config['eplus_idf_filename'],
                  self.config['weather_filename'],
                  self.config['eplus_output_path'],
                  callback_function_bp),
            daemon=True,
        )
        _eplus_thread.start()
        self._wait_for_obs()

    def _wait_for_obs(self, timeout_s: float = 30.0):
        deadline = time.monotonic() + timeout_s
        while not obs_event.is_set():
            if _eplus_failed:
                logger.error("EnergyPlus failed — aborting wait")
                return
            if time.monotonic() > deadline:
                logger.error("Timed out waiting for EnergyPlus observation")
                return
            obs_event.wait(timeout=0.05)
        obs_event.clear()

    def _advance_one_step(self):
        act_event.set()
        self._wait_for_obs()

    def _skip_timesteps_until_workday(self):
        while eplus_sim_step < self.config['total_steps'] and not _eplus_failed:
            next_dt = get_time(eplus_data_collection[-1]) + datetime.timedelta(minutes=1)
            if next_dt.weekday() <= 4:
                break
            self._advance_one_step()

    def calc_state(self, variables: Dict[str, Any]) -> np.ndarray:
        """Convert augmented E+ variables into an ordered observation array."""
        obs_dict = get_observations(variables, self.config)
        obs_vec = [obs_dict[k] for k in self.config['observations'] if k in obs_dict]
        return np.array(obs_vec, dtype=np.float32)

    def calc_reward(self, variables: Dict[str, Any]) -> float:
        return get_reward(variables, self._get_outdoor_rolling())

    # ----- Gymnasium API ---------------------------------------------------

    def reset(self, seed=None, options=None):
        logger.info('---------- reset -----------')
        super().reset(seed=seed)

        if not _eplus_running or eplus_sim_step >= self.config['total_steps']:
            self._start_energyplus()
        else:
            logger.info('eplus_sim_step %s — continuing current E+ run', eplus_sim_step)

        if self.config.get('skip_weekends') == 1 and not _eplus_failed:
            self._skip_timesteps_until_workday()
            if eplus_sim_step >= self.config['total_steps']:
                self._start_energyplus()

        if _eplus_failed or len(eplus_data_collection) == 0:
            logger.error("No EnergyPlus data — returning zeros")
            return np.zeros(len(self.config['observations']), dtype=np.float32), \
                   {"eplus_failed": True}

        curr = self._prepare_step_data(eplus_data_collection[-1])
        observations = self.calc_state(curr)
        self.rl_data_collection.append({**curr, **get_observations(curr, self.config)})
        return observations, {}

    def step(self, action):
        action_dict = dict(zip(self.config['rl_actions'], action))

        if self.config.get('actuator_normalize') == 1:
            for k in self.config['rl_actions']:
                lo, hi = self.config['action_range'][k]
                action_dict[k] = (action_dict[k] + 1.0) / 2.0 * (hi - lo) + lo

        for k in self.config['rl_actions']:
            lo, hi = self.config['action_range'][k]
            action_dict[k] = float(np.clip(action_dict[k], lo, hi))

        if 'heating_setpoint' in action_dict and 'cooling_setpoint' in action_dict:
            if action_dict['heating_setpoint'] > action_dict['cooling_setpoint']:
                action_dict['heating_setpoint'] = round(action_dict['cooling_setpoint'], 2) - 0.02

        actions_list.append(action_dict)
        self._advance_one_step()

        if _eplus_failed or len(eplus_data_collection) == 0:
            logger.error("EnergyPlus failed during step — truncating episode")
            return (np.zeros(len(self.config['observations']), dtype=np.float32),
                    0.0, True, True, {"eplus_failed": True})

        if eplus_sim_step >= self.config['total_steps']:
            done = True
            truncated = True
        else:
            done = False
            truncated = False

        curr = self._prepare_step_data(eplus_data_collection[-1])
        observations = self.calc_state(curr)
        reward = self.calc_reward(curr)

        self.rl_data_collection.append(
            {**curr, **get_observations(curr, self.config), **action_dict,
             'reward': reward, 'eplus_sim_step': eplus_sim_step})

        return observations, reward, done, truncated, {}

    def close(self):
        global _eplus_thread
        logger.info('Closing EnergyPlusEnv')
        stop_event.set()
        act_event.set()
        if _eplus_thread is not None and _eplus_thread.is_alive():
            _eplus_thread.join(timeout=30)
        _eplus_thread = None
        super().close()
