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
import datetime
import time
import threading
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

# Inter-thread synchronisation events
act_event  = threading.Event()   # RL  → E+ : "action is ready, proceed"
obs_event  = threading.Event()   # E+  → RL : "observations are ready"
stop_event = threading.Event()   # RL  → E+ : "stop blocking, shut down"

# Timestep data written by the E+ callback, consumed by the Gym env
eplus_data_collection = []       # List[Dict] — one entry per simulated timestep
actions_list = []                # List[Dict] — actions sent by the RL agent
eplus_sim_step = 0               # Running count of simulated timesteps

# Simulation lifecycle
_eplus_running = False
_eplus_failed  = False
_eplus_thread  = None            # Reference to the daemon thread running E+
_ep_state      = None            # Opaque E+ state pointer
_run_counter   = 0               # Monotonic counter for unique output dirs

# E+ API handles (resolved once per simulation run)
_handles_initialized = False
_sensor_handles: Dict[str, int] = {}    # alias → variable handle
_actuator_handles: Dict[str, int] = {}  # alias → actuator handle
_meter_handles: Dict[str, int] = {}     # alias → meter handle

# Conservative fallback action applied during warm-up and early timesteps
# to avoid destabilising the simulation before the agent is active.
SAFE_INITIAL_ACTION = {
    "cooling_setpoint": 21.5,    # °C
    "heating_setpoint": 21.5,    # °C
    "ahu_supply_temp": 19.0,     # °C
    "supply_fan_flow": 0.20,     # m³/s
}

# Number of post-warmup timesteps that still use SAFE_INITIAL_ACTION
# to allow the model to reach a numerically stable state.
POST_WARMUP_HOLD_STEPS = 8
_post_warmup_steps = 0
_warmup_complete_seen = False


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_get(dic: dict, key: str, default=0.0):
    """Return ``dic[key]`` if present, else ``default``. Avoids KeyError."""
    return dic[key] if key in dic else default


# ---------------------------------------------------------------------------
# E+ data-exchange helpers (called from the callback thread)
# ---------------------------------------------------------------------------

def _init_handles(state) -> bool:
    """Resolve E+ variable, meter, and actuator handles.

    Called on every timestep but performs work only once (guarded by
    ``_handles_initialized``).  Returns False if the E+ data layer is
    not yet ready, signalling the caller to retry next timestep.
    """
    global _handles_initialized, _sensor_handles, _actuator_handles, _meter_handles

    if _handles_initialized:
        return True

    exch = api.exchange

    if not exch.api_data_fully_ready(state):
        return False  # IDF not fully parsed yet; try again next timestep

    _meter_handles = {}
    _sensor_handles = {}
    _actuator_handles = {}

    # Resolve handles from definitions in variable_config.py
    for alias, var_name, key_value, _ in SENSOR_DEF:
        _sensor_handles[alias] = exch.get_variable_handle(state, var_name, key_value)

    for alias, meter_name in METER_DEF:
        _meter_handles[alias] = exch.get_meter_handle(state, meter_name)

    for alias, comp_type, control_type, key_value in ACTUATOR_DEF:
        _actuator_handles[alias] = exch.get_actuator_handle(state, comp_type, control_type, key_value)

    # Handle value of -1 indicates an unresolved reference (typo or missing object)
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
    """Collect all sensor and meter values for the current E+ timestep.

    Returns a flat dict keyed by alias (e.g. ``outdoor_temp``, ``electricity_hvac``).
    Time fields (year, month, …) are included for downstream conversion.
    """
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
    """Write ``action_dict`` values into the corresponding E+ actuators."""
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
    """Per-timestep callback registered with EnergyPlus.

    Execution flow each timestep:

    1. Read current sensor/meter values (observations).
    2. Signal ``obs_event`` so the RL thread can consume them.
    3. Block on ``act_event`` until the RL thread provides the next action.
    4. Select the appropriate action (safe default during warm-up,
       RL-chosen action otherwise) and apply it to the actuators.
    """
    global eplus_sim_step, _post_warmup_steps, _warmup_complete_seen

    try:
        if not _init_handles(state):
            return

        # 1. Collect observations
        curr = _read_current_timestep(state)
        eplus_data_collection.append(curr)
        eplus_sim_step += 1

        # 2. Notify RL thread
        obs_event.set()

        # 3. Block until RL thread signals its action
        while not act_event.is_set():
            if stop_event.is_set():
                return
            act_event.wait(timeout=0.1)
        act_event.clear()

        # 4. Determine action to apply
        in_warmup = api.exchange.warmup_flag(state)

        if in_warmup:
            # Warm-up phase: use conservative setpoints to avoid instability
            action_to_apply = SAFE_INITIAL_ACTION
        else:
            if not _warmup_complete_seen:
                _warmup_complete_seen = True
                _post_warmup_steps = 0

            if _post_warmup_steps < POST_WARMUP_HOLD_STEPS:
                # Transition buffer: keep safe defaults a few more steps
                action_to_apply = SAFE_INITIAL_ACTION
                _post_warmup_steps += 1
            elif len(actions_list) > 0 and actions_list[-1] is not None:
                action_to_apply = actions_list[-1]
            else:
                action_to_apply = SAFE_INITIAL_ACTION

        _apply_action(state, action_to_apply)

    except Exception as exc:
        logger.exception("Error in EnergyPlus callback: %s", exc)
        obs_event.set()  # Ensure RL thread is never left blocking


# ---------------------------------------------------------------------------
# E+ simulation launcher (runs inside daemon thread)
# ---------------------------------------------------------------------------

def run_energyplus(idf_file: str, weather_file: str, output_path: str, callback) -> int:
    """Execute a full EnergyPlus simulation (blocking).

    Intended to be called from a daemon thread.  Registers ``callback``
    at the ``BeginZoneTimestepAfterInitHeatBalance`` calling point, then
    runs E+ to completion.

    Returns 0 on success, non-zero on failure.
    """
    global _ep_state, _handles_initialized, eplus_sim_step, _eplus_running, _eplus_failed, _run_counter

    logger.info("Starting EnergyPlus simulation")
    logger.info("IDF: %s", idf_file)
    logger.info("Weather: %s", weather_file)
    logger.info("Output path: %s", output_path)

    if not os.path.isfile(weather_file):
        logger.error("Weather file does not exist: %s", weather_file)
        _eplus_failed = True
        _eplus_running = False
        obs_event.set()       # unblock RL if it is waiting
        return 1

    _handles_initialized = False
    eplus_sim_step = 0
    _eplus_running = True
    _eplus_failed = False

    # Each run writes to a unique subdirectory (run_1, run_2, …) to avoid
    # SQLite / file-lock conflicts when training spans multiple episodes.
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
    obs_event.set()  # Unblock RL thread if it is still waiting

    # Release the E+ state to free native memory
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
    """Convert an E+ timestep dict to ``datetime.datetime``.

    Handles E+ conventions: hour may be 24 (midnight rollover) and
    minutes may be 60 (end-of-hour marker).
    """
    year = int(_safe_get(step_data, "year", 2001))
    month = int(_safe_get(step_data, "month", 1))
    day = int(_safe_get(step_data, "day", 1))
    hour = int(_safe_get(step_data, "hour", 0))
    minute = int(_safe_get(step_data, "minutes", 0))

    # EnergyPlus minutes can be 60 (end of hour) — normalize
    if minute >= 60:
        minute = 0
        hour += 1

    # EnergyPlus hour can reach 24 (midnight next day) — normalize
    if hour >= 24:
        hour = 0

    return datetime.datetime(year, month, day, hour, minute)


def get_observations(step_data: Dict[str, Any], config) -> Dict[str, float]:
    """Extract the RL observation vector from raw E+ timestep data.

    Returns a dict whose keys match ``config['observations']``.
    When ``config['observation_normalize'] == 1``, values are min–max
    scaled to [-1, 1] using the bounds in ``config['rl_observation_min/max']``.
    """
    obs_keys = config["observations"]

    # Build obs dict dynamically from whatever keys are listed in config
    obs = {k: float(_safe_get(step_data, k, 0.0)) for k in obs_keys}

    # Optional min–max normalisation to [-1, 1]
    if config.get('observation_normalize') == 1:
        obs_min = config['rl_observation_min']
        obs_max = config['rl_observation_max']
        for i, key in enumerate(obs_keys):
            lo, hi = obs_min[i], obs_max[i]
            obs[key] = float(2.0 * (np.clip(obs[key], lo, hi) - lo) / (hi - lo) - 1.0)

    return obs


def get_reward(step_data: Dict[str, Any]) -> float:
    """Compute the scalar reward signal for the current timestep.

    Mirrors the hackathon cost function (negated, since RL maximises reward):
        - Energy cost: electricity 0.11 €/kWh, gas 0.06 €/kWh
        - CO2 penalty per zone: >770 ppm → 2 €/h, >970 → 10, >1220 → 50
        - Temperature penalty per zone: outside S1 → 1 €/h, S2 → 5, S3 → 25

    All costs are per 15-min timestep (÷4 from hourly rates).
    """
    TIMESTEP_HOURS = 0.25  # 15 minutes

    # --- Energy cost ---
    elec_j = _safe_get(step_data, "electricity_hvac", 0.0)
    gas_j = _safe_get(step_data, "gas_total", 0.0)
    elec_kwh = elec_j / 3_600_000.0
    gas_kwh = gas_j / 3_600_000.0
    energy_cost = elec_kwh * 0.11 + gas_kwh * 0.06

    # --- CO2 penalty (per zone) ---
    co2_cost = 0.0
    for i in range(1, 6):
        co2 = _safe_get(step_data, f"space{i}_co2", 400.0)
        if co2 > 1220:
            co2_cost += 50.0 * TIMESTEP_HOURS
        elif co2 > 970:
            co2_cost += 10.0 * TIMESTEP_HOURS
        elif co2 > 770:
            co2_cost += 2.0 * TIMESTEP_HOURS

    # --- Temperature penalty (per zone, simplified S-class bands) ---
    # Finnish S1/S2/S3 comfort bands depend on 24h rolling outdoor temp.
    # Simplified: target ~21-23°C, S1 ±1°C, S2 ±2°C, S3 ±3°C
    temp_cost = 0.0
    target = 22.0
    for i in range(1, 6):
        temp = _safe_get(step_data, f"space{i}_temp", 22.0)
        dev = abs(temp - target)
        if dev > 3.0:
            temp_cost += 25.0 * TIMESTEP_HOURS
        elif dev > 2.0:
            temp_cost += 5.0 * TIMESTEP_HOURS
        elif dev > 1.0:
            temp_cost += 1.0 * TIMESTEP_HOURS

    total_cost = energy_cost + co2_cost + temp_cost
    return -total_cost


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------

class EnergyPlusEnv(gym.Env):
    """Gymnasium-compatible wrapper around EnergyPlus.

    E+ runs in a daemon thread; this class exposes the standard
    ``reset()`` / ``step(action)`` / ``close()`` interface expected
    by RL training loops (e.g. Stable-Baselines3).

    Thread synchronisation per timestep::

        E+ callback (daemon)              RL agent (main thread)
        ────────────────────              ──────────────────────
        collect observations
        obs_event.set()  ──────────────>  wait(obs_event)  [reset / step]
        wait(act_event)  <──────────────  act_event.set()  [step]
        apply action
        …next timestep…
    """

    def __init__(self, config):
        """Initialise action/observation spaces from ``config``."""
        super().__init__()
        self.config = config

        # Action space: normalised [-1, 1] or physical units
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

        # Observation space: normalised [-1, 1] or physical units
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

        # Accumulated per-step records for post-episode analysis
        self.rl_data_collection = []

    def _start_energyplus(self):
        """(Re-)launch EnergyPlus in a daemon thread.

        Joins any previously running thread, resets shared state, starts
        a new simulation, and blocks until the first observation arrives.
        """
        global _eplus_thread, _eplus_failed
        global _post_warmup_steps, _warmup_complete_seen
        _post_warmup_steps = 0
        _warmup_complete_seen = False

        # Gracefully terminate previous run (if any)
        stop_event.set()
        act_event.set()
        if _eplus_thread is not None and _eplus_thread.is_alive():
            logger.info("Joining previous EnergyPlus thread …")
            _eplus_thread.join(timeout=30)

        # Reset shared state
        eplus_data_collection.clear()
        actions_list.clear()
        obs_event.clear()
        act_event.clear()
        stop_event.clear()
        _eplus_failed = False

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

        # Wait for the first timestep observation from E+
        self._wait_for_obs()

    def _wait_for_obs(self, timeout_s: float = 30.0):
        """Block until ``obs_event`` is set or a failure / timeout occurs."""
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
        """Unblock the E+ callback (``act_event``) and wait for the next observation."""
        act_event.set()
        self._wait_for_obs()

    def _skip_timesteps_until_workday(self):
        """Advance through weekend timesteps (Sat/Sun) without RL interaction."""
        while eplus_sim_step < self.config['total_steps'] and not _eplus_failed:
            next_dt = get_time(eplus_data_collection[-1]) + datetime.timedelta(minutes=1)
            if next_dt.weekday() <= 4:       # Mon–Fri
                break
            self._advance_one_step()

    def calc_state(self, variables: Dict[str, Any]) -> np.ndarray:
        """Convert raw E+ variables into an ordered observation array."""
        obs_dict = get_observations(variables, self.config)
        obs_vec = [
            obs_dict[k] for k in self.config['observations'] if k in obs_dict
        ]
        return np.array(obs_vec, dtype=np.float32)

    @staticmethod
    def calc_reward(variables: Dict[str, Any]) -> float:
        """Delegate to the module-level ``get_reward`` function."""
        return get_reward(variables)

    def reset(self, seed=None, options=None):
        """Start a new episode or continue the current E+ simulation.

        Returns ``(observations, info)`` per the Gymnasium API.
        """
        logger.info('---------- reset -----------')
        super().reset(seed=seed)

        if not _eplus_running or eplus_sim_step >= self.config['total_steps']:
            # First call, or previous E+ case finished → start new case
            self._start_energyplus()
        else:
            logger.info('eplus_sim_step %s — continuing current E+ run', eplus_sim_step)

        # Optionally skip weekend timesteps
        if self.config.get('skip_weekends') == 1 and not _eplus_failed:
            self._skip_timesteps_until_workday()
            # If skipping consumed all steps, restart
            if eplus_sim_step >= self.config['total_steps']:
                self._start_energyplus()

        # Failure guard
        if _eplus_failed or len(eplus_data_collection) == 0:
            logger.error("No EnergyPlus data — returning zeros")
            return np.zeros(len(self.config['observations']), dtype=np.float32), \
                   {"eplus_failed": True}

        curr = eplus_data_collection[-1]
        observations = self.calc_state(curr)
        self.rl_data_collection.append({**curr, **get_observations(curr, self.config)})
        return observations, {}

    def step(self, action):
        """Apply *action*, advance E+ one timestep, return Gymnasium 5-tuple."""

        # Map array → named dict
        action_dict = dict(zip(self.config['rl_actions'], action))

        # Denormalise from [-1, 1] to physical units
        if self.config.get('actuator_normalize') == 1:
            for k in self.config['rl_actions']:
                lo, hi = self.config['action_range'][k]
                action_dict[k] = (action_dict[k] + 1.0) / 2.0 * (hi - lo) + lo

        # Hard-clip to actuator limits
        for k in self.config['rl_actions']:
            lo, hi = self.config['action_range'][k]
            action_dict[k] = float(np.clip(action_dict[k], lo, hi))

        # Enforce heating ≤ cooling constraint (round to 2 dp to satisfy E+ parser)
        if 'heating_setpoint' in action_dict and 'cooling_setpoint' in action_dict:
            # If the heating setpoint exceeds the cooling setpoint, the EnergyPlus simulation will fail
            if action_dict['heating_setpoint'] > action_dict['cooling_setpoint']:
                action_dict['heating_setpoint'] = round(action_dict['cooling_setpoint'], 2) - 0.02


        # Publish action for the E+ callback
        actions_list.append(action_dict)

        # Advance E+ by one timestep
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

        curr = eplus_data_collection[-1]
        observations = self.calc_state(curr)
        reward = self.calc_reward(curr)

        self.rl_data_collection.append(
            {**curr, **get_observations(curr, self.config), **action_dict,
             'reward': reward, 'eplus_sim_step': eplus_sim_step})

        return observations, reward, done, truncated, {}

    def close(self):
        """Signal the E+ thread to terminate and join it."""
        global _eplus_thread
        logger.info('Closing EnergyPlusEnv')
        stop_event.set()
        act_event.set()  # Unblock callback if it is waiting on an action
        if _eplus_thread is not None and _eplus_thread.is_alive():
            _eplus_thread.join(timeout=30)
        _eplus_thread = None
        super().close()
