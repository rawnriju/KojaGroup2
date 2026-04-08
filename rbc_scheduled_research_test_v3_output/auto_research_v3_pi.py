#!/usr/bin/env python3
"""Auto-research script for RBC scheduled controller optimization.

Runs EnergyPlus with parameterized RBC, computes total_cost_eur, prints breakdown.
Edit the PARAMS dict to tune, then re-run.
"""

import sys
import os
import shutil
import json
import time
from pathlib import Path

# --- EnergyPlus Python API setup ---
ENERGYPLUS_DIR = r"C:\EnergyPlusV25-2-0"
sys.path.append(ENERGYPLUS_DIR)
from pyenergyplus.api import EnergyPlusAPI

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
IDF_FILE = SCRIPT_DIR.parent / "DOAS_wNeutralSupplyAir_wFanCoilUnits.idf"
EPW_FILE = SCRIPT_DIR.parent / "FIN_TR_Tampere.Satakunnankatu.027440_TMYx.2004-2018.epw"
OUT_DIR = SCRIPT_DIR / "eplus_out"

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# TUNABLE PARAMETERS — edit these to experiment
# ═══════════════════════════════════════════════════════════════════════════

PARAMS = {
    # Zone heating setpoint [°C] — linear ramp between outdoor temp bounds
    # S1 lower band: 20.5°C at t_out<=0, 20.5+0.075*t at 0<t<20, 22°C at t>=20
    # We want to stay above S1 lower → heat a bit above it
    "htg_setpoint_low": 21.2,     # heating SP at outdoor <= outdoor_comp_low (S1 lower=20.5, +0.7 margin)
    "htg_setpoint_high": 22.4,    # heating SP at outdoor >= outdoor_comp_high (S1 lower=22.0, +0.4 margin)

    # Zone cooling setpoint [°C]
    # S1 upper band: 22.0°C at t_out<=0, 22.5+0.166*t at 0<t<15, 25°C at t>=15
    # We want to stay below S1 upper — allow slightly more drift to save energy
    "clg_setpoint_low": 21.9,     # cooling SP at outdoor <= outdoor_comp_low (S1 upper=22.0, -0.1 margin)
    "clg_setpoint_high": 24.7,    # cooling SP at outdoor >= outdoor_comp_high (S1 upper=25.0, -0.3 margin)

    # Outdoor temp compensation range
    "outdoor_comp_low": 0.0,
    "outdoor_comp_high": 20.0,

    # Supply air temperature compensation (from return air / plenum temp)
    "return_air_temp_low": 21.0,
    "return_air_temp_high": 24.5,
    "sup_temp_at_low": 17.0,      # optimal supply temp curve
    "sup_temp_at_high": 16.0,     # min allowed supply temp

    # CO2 demand-controlled ventilation
    "co2_min_limit": 650,
    "co2_max_limit": 760,

    # Cold outdoor temp flow limiting
    "outdoor_temp_low_limit": -30.0,
    "outdoor_temp_high_limit": -20.0,

    # Fan flow rates [kg/s]
    "flow_off": 0.0,
    "flow_low": 0.05,             # minimal base flow
    "flow_moderate": 0.30,
    "flow_boost": 1.0,
    "flow_max": 1.0,

    # Schedule hours (workday)
    "nightflush_start": 0.0,      # disabled (start==end)
    "nightflush_end": 0.0,
    "pre_work_flush_start": 0.0,   # disabled
    "pre_work_flush_end": 0.0,
    "working_hours_start": 5.5,
    "working_hours_end": 20.0,    # end earlier to save energy

    # Weekend schedule
    "weekend_nightflush_start": 0.0,
    "weekend_nightflush_end": 0.0,   # disabled
    "weekend_flush_start": 0.0,      # disabled
    "weekend_flush_end": 0.0,

    # Solar gain pre-cooling: if direct solar > threshold, lower cooling SP
    "solar_precool_enabled": False,
    "solar_precool_threshold": 200.0,  # W/m2 direct solar — react earlier
    "solar_precool_clg_offset": -0.8,  # °C offset to cooling setpoint

    # S1 band tracking margins (used by zone_setpoints_s1)
    "s1_htg_margin": 0.55,        # °C above S1 lower bound for heating SP
    "s1_clg_margin": 0.05,        # °C below S1 upper bound

    # Wind chill compensation: if windy + cold, raise heating SP slightly
    "wind_comp_enabled": False,
    "wind_speed_threshold": 4.0,       # m/s
    "wind_cold_outdoor_threshold": 5.0, # only when outdoor < this
    "wind_htg_offset": 0.5,            # °C added to heating SP
}

# ═══════════════════════════════════════════════════════════════════════════
# Variable config (sensors, actuators, meters)
# ═══════════════════════════════════════════════════════════════════════════

ZONE_NAMES = [
    "SPACE1-1 THERMAL ZONE", "SPACE2-1 THERMAL ZONE", "SPACE3-1 THERMAL ZONE",
    "SPACE4-1 THERMAL ZONE", "SPACE5-1 THERMAL ZONE",
]

VARIABLES = {
    "outdoor_temp": ("Site Outdoor Air Drybulb Temperature", "Environment"),
    "outdoor_rh": ("Site Outdoor Air Relative Humidity", "Environment"),
    "direct_solar": ("Site Direct Solar Radiation Rate per Area", "Environment"),
    "diffuse_solar": ("Site Diffuse Solar Radiation Rate per Area", "Environment"),
    "wind_speed": ("Site Wind Speed", "Environment"),
    "wind_direction": ("Site Wind Direction", "Environment"),
    "sky_temp": ("Site Sky Temperature", "Environment"),
    "plenum_temp": ("Zone Air Temperature", "PLENUM-1 THERMAL ZONE"),
    "co2_plenum": ("Zone Air CO2 Concentration", "PLENUM-1 THERMAL ZONE"),
}
for i, zone in enumerate(ZONE_NAMES, 1):
    VARIABLES[f"space{i}_temp"] = ("Zone Air Temperature", zone)
    VARIABLES[f"space{i}_rh"] = ("Zone Air Relative Humidity", zone)
    VARIABLES[f"space{i}_occupancy"] = ("Zone People Occupant Count", zone)
    VARIABLES[f"space{i}_co2"] = ("Zone Air CO2 Concentration", zone)

ACTUATORS = {
    "clg_setpoint": ("Schedule:Compact", "Schedule Value", "CLG-SETP-SCH"),
    "htg_setpoint": ("Schedule:Compact", "Schedule Value", "HTG-SETP-SCH"),
    "ahu_temperature_setpoint": ("Schedule:Compact", "Schedule Value", "AHU_Supply_Temp_Schedule"),
    "ahu_mass_flow_rate_setpoint": ("Fan", "Fan Air Mass Flow Rate", "DOAS SYSTEM SUPPLY FAN"),
}

METERS = {
    "electricity_hvac": "Electricity:HVAC",
    "gas_total": "NaturalGas:Facility",
    "fans_electricity": "Fans:Electricity",
}


# ═══════════════════════════════════════════════════════════════════════════
# RBC Model (parameterized)
# ═══════════════════════════════════════════════════════════════════════════

class PIController:
    """Simple PI controller with anti-windup clamping."""
    def __init__(self, kp, ki, out_min, out_max):
        self.kp = kp
        self.ki = ki
        self.out_min = out_min
        self.out_max = out_max
        self.integral = 0.0

    def step(self, error, dt=1.0):
        self.integral += error * dt
        # Anti-windup: clamp integral
        max_i = (self.out_max - self.out_min) / max(abs(self.ki), 1e-6) * 0.5
        self.integral = np.clip(self.integral, -max_i, max_i)
        output = self.kp * error + self.ki * self.integral
        return float(np.clip(output, self.out_min, self.out_max))


class RBCModel:
    def __init__(self, params):
        self.p = params
        self._outdoor_temps = []
        self._outdoor_24h_avg = None

        # PI controllers for supply air temp (error = zone_temp - target)
        # Positive error = too warm → lower supply temp
        self.pi_supply = PIController(
            kp=-0.8, ki=-0.05, out_min=16.0, out_max=21.0
        )

        # PI controller for fan flow based on CO2
        # Error = co2 - target → positive = too much CO2 → more flow
        self.pi_co2_flow = PIController(
            kp=0.010, ki=0.001, out_min=0.0, out_max=1.0
        )

    def _update_outdoor_avg(self, outdoor_temp):
        self._outdoor_temps.append(outdoor_temp)
        if len(self._outdoor_temps) > 96:
            self._outdoor_temps = self._outdoor_temps[-96:]
        self._outdoor_24h_avg = sum(self._outdoor_temps) / len(self._outdoor_temps)

    def _s1_bands(self):
        t = self._outdoor_24h_avg if self._outdoor_24h_avg is not None else 0.0
        if t <= 0:
            lower, upper = 20.5, 22.0
        elif t <= 15:
            lower = 20.5 + 0.075 * t
            upper = 22.5 + 0.166 * t
        elif t <= 20:
            lower = 20.5 + 0.075 * t
            upper = 25.0
        else:
            lower, upper = 22.0, 25.0
        return lower, upper

    def calculate_setpoints(self, zone_temps, outdoor_temp, return_air_temp, occupancy,
                            hour, day, zone_co2s, direct_solar=0.0, wind_speed=0.0,
                            outdoor_rh=50.0, diffuse_solar=0.0):
        """
        New signature: takes per-zone temps and co2s (lists of 5) instead of averages.
        This allows per-zone-aware control decisions.
        """
        p = self.p
        self._update_outdoor_avg(outdoor_temp)
        lower_s1, upper_s1 = self._s1_bands()

        # ── Per-zone awareness: target the WORST zone ──
        # Find the zone closest to violating S1 boundaries
        min_zone_temp = min(zone_temps)
        max_zone_temp = max(zone_temps)
        avg_zone_temp = sum(zone_temps) / len(zone_temps)
        max_co2 = max(zone_co2s)

        # S1 band target: the point that maximizes distance from both boundaries
        # Weighted slightly toward upper to minimize heating (dominant cost in Finland)
        s1_mid = (lower_s1 + upper_s1) / 2.0
        target = s1_mid + p.get("target_offset", 0.5)  # bias warm to reduce heating

        # ── SETPOINTS from S1 bands ──
        # Adaptive margins based on S1 band width
        s1_width = upper_s1 - lower_s1  # 1.5°C in cold, 3°C in warm
        # More margin when band is narrow (cold), less when wide (warm) — proportional
        base_htg = p.get("s1_htg_margin", 0.55)
        base_clg = p.get("s1_clg_margin", 0.05)
        # Scale: at width=1.5, multiply by 1.0; at width=3.0, multiply by 0.6
        width_factor = np.clip(1.0 - 0.20 * (s1_width - 1.5), 0.5, 1.0)
        htg = lower_s1 + base_htg * width_factor
        clg = upper_s1 - base_clg

        # Per-zone margins for economizer check
        cold_margin = min_zone_temp - lower_s1
        hot_margin = upper_s1 - max_zone_temp

        if htg > clg:
            htg = clg - 0.3

        # ── SUPPLY AIR TEMP — fixed curve (proven optimal) ──
        # Return air compensation: lower supply temp when return air is warm
        p = self.p
        rt_low, rt_high = p["return_air_temp_low"], p["return_air_temp_high"]
        st_low, st_high = p["sup_temp_at_low"], p["sup_temp_at_high"]
        if return_air_temp <= rt_low:
            supply_air_temp = st_low
        elif return_air_temp >= rt_high:
            supply_air_temp = st_high
        else:
            slope = (st_high - st_low) / (rt_high - rt_low)
            supply_air_temp = float(np.clip(st_low + slope * (return_air_temp - rt_low),
                                            min(st_low, st_high), max(st_low, st_high)))

        # Solar compensation disabled — not helpful in Finland
        total_solar = direct_solar + diffuse_solar

        # ── ENTHALPY ECONOMIZER ──
        # When outdoor air is cool and dry enough, it's free cooling
        # Outdoor enthalpy estimate: h ≈ 1.006*T + (W * 2501)
        # Simplified: if outdoor is cooler than zone and zone needs cooling, boost ventilation
        free_cooling_available = (outdoor_temp < avg_zone_temp - 1.0 and
                                  avg_zone_temp > s1_mid and
                                  outdoor_temp > 5.0)  # don't use when too cold (heating cost)

        # ── FAN FLOW: threshold-based CO2 DCV (proven optimal) + enthalpy economizer ──
        co2_min = p["co2_min_limit"]
        co2_max = p["co2_max_limit"]
        if max_co2 <= co2_min:
            co2_flow = 0.0
        elif max_co2 >= co2_max:
            co2_flow = p["flow_boost"]
        else:
            fraction = (max_co2 - co2_min) / (co2_max - co2_min)
            co2_flow = p["flow_low"] + fraction * (p["flow_boost"] - p["flow_low"])

        # Schedule-based minimum flow
        is_workday = day != 1 and day != 7
        is_working = is_workday and p["working_hours_start"] <= hour < p["working_hours_end"]
        min_flow = p["flow_low"] if is_working else 0.0

        flow = max(co2_flow, min_flow)

        # Enthalpy economizer disabled — testing
        # if free_cooling_available and hot_margin < 0.8:
        #     flow = max(flow, 0.3)

        flow = float(np.clip(flow, 0.0, 1.0))

        return htg, clg, supply_air_temp, flow


# ═══════════════════════════════════════════════════════════════════════════
# Controller
# ═══════════════════════════════════════════════════════════════════════════

class Controller:
    def __init__(self, api, model):
        self.api = api
        self.model = model
        self.handles = {}

    def initialize_handles(self, state):
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

    def get_variable(self, name, state, default=0.0):
        handle = self.handles.get(name)
        if handle is None or handle == -1:
            return default
        if name in METERS:
            return self.api.exchange.get_meter_value(state, handle)
        return self.api.exchange.get_variable_value(state, handle)

    def set_actuator(self, name, value, state):
        handle = self.handles.get(name)
        if handle is not None and handle != -1:
            self.api.exchange.set_actuator_value(state, handle, value)

    def control_callback(self, state):
        if not self.handles:
            return

        outdoor_temp = self.get_variable("outdoor_temp", state)
        plenum_temp = self.get_variable("plenum_temp", state)
        direct_solar = self.get_variable("direct_solar", state)
        diffuse_solar = self.get_variable("diffuse_solar", state)
        wind_speed = self.get_variable("wind_speed", state)
        outdoor_rh = self.get_variable("outdoor_rh", state)
        hour = float(self.api.exchange.hour(state))
        day = float(self.api.exchange.day_of_week(state))

        zone_temps, zone_co2s, occs = [], [], []
        for i in range(1, 6):
            zone_temps.append(self.get_variable(f"space{i}_temp", state))
            zone_co2s.append(self.get_variable(f"space{i}_co2", state))
            occs.append(self.get_variable(f"space{i}_occupancy", state))

        total_occupancy = sum(occs)

        htg, clg, supply_air_temp, flow = self.model.calculate_setpoints(
            zone_temps=zone_temps,
            outdoor_temp=outdoor_temp,
            return_air_temp=plenum_temp,
            occupancy=total_occupancy,
            hour=hour,
            day=day,
            zone_co2s=zone_co2s,
            direct_solar=direct_solar,
            wind_speed=wind_speed,
            outdoor_rh=outdoor_rh,
            diffuse_solar=diffuse_solar,
        )

        self.set_actuator("htg_setpoint", htg, state)
        self.set_actuator("clg_setpoint", clg, state)
        self.set_actuator("ahu_temperature_setpoint", supply_air_temp, state)
        self.set_actuator("ahu_mass_flow_rate_setpoint", flow, state)


# ═══════════════════════════════════════════════════════════════════════════
# Run simulation + compute cost
# ═══════════════════════════════════════════════════════════════════════════

def run_simulation(params):
    """Run EnergyPlus with given params, return cost dict."""
    # Clean output dir
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    api = EnergyPlusAPI()
    state = api.state_manager.new_state()

    # Request weather variables that aren't in the IDF Output:Variable list
    for name, (var, key) in VARIABLES.items():
        api.exchange.request_variable(state, var, key)

    model = RBCModel(params)
    controller = Controller(api, model)

    api.runtime.callback_after_new_environment_warmup_complete(
        state, controller.initialize_handles
    )
    api.runtime.callback_begin_zone_timestep_after_init_heat_balance(
        state, controller.control_callback
    )

    args = ["-d", str(OUT_DIR), "-w", str(EPW_FILE), "-r", str(IDF_FILE)]
    print(f"Running EnergyPlus...")
    t0 = time.time()
    rc = api.runtime.run_energyplus(state, args)
    elapsed = time.time() - t0
    print(f"EnergyPlus finished in {elapsed:.1f}s with code {rc}")

    if rc != 0:
        print("SIMULATION FAILED!")
        return None

    # Compute cost
    from cost_calculator import load_eplusout, compute_total_cost
    csv_path = OUT_DIR / "eplusout.csv"
    df = load_eplusout(str(csv_path))
    costs = compute_total_cost(df)
    return costs


def print_costs(costs, label=""):
    if costs is None:
        print(f"[{label}] FAILED - no cost data")
        return
    print(f"\n{'='*60}")
    if label:
        print(f"  {label}")
    print(f"  Energy:  {costs['energy_cost_eur']:>10.2f} EUR")
    print(f"  CO2:     {costs['co2_penalty_eur']:>10.2f} EUR")
    print(f"  Temp:    {costs['temp_penalty_eur']:>10.2f} EUR")
    print(f"  TOTAL:   {costs['total_cost_eur']:>10.2f} EUR")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    costs = run_simulation(PARAMS)
    print_costs(costs, "Current RBC Parameters")
