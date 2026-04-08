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

class RBCModel:
    def __init__(self, params):
        self.p = params
        # Rolling outdoor temp buffer for 24h average (96 steps at 15min)
        self._outdoor_temps = []
        self._outdoor_24h_avg = None

    def co2_flow_control(self, hour, day, co2_concentration, outdoor_temp):
        p = self.p
        is_workday = day != 1 and day != 7

        is_nightflush = is_workday and p["nightflush_start"] <= hour < p["nightflush_end"]
        is_nightflush_weekend = not is_workday and p["weekend_nightflush_start"] <= hour < p["weekend_nightflush_end"]
        is_before_work_flush = is_workday and p["pre_work_flush_start"] <= hour < p["pre_work_flush_end"]
        is_weekend_flush = not is_workday and p["weekend_flush_start"] <= hour < p["weekend_flush_end"]
        is_working_hours = is_workday and p["working_hours_start"] <= hour < p["working_hours_end"]

        base_flow = p["flow_off"]
        if is_before_work_flush:
            base_flow = p["flow_moderate"]
        elif is_nightflush or is_nightflush_weekend or is_weekend_flush or is_working_hours:
            base_flow = p["flow_low"]

        # CO2 demand control
        co2_flow = 0.0
        if co2_concentration <= p["co2_min_limit"]:
            co2_flow = 0.0
        elif co2_concentration >= p["co2_max_limit"]:
            co2_flow = p["flow_boost"]
        else:
            fraction = (co2_concentration - p["co2_min_limit"]) / (p["co2_max_limit"] - p["co2_min_limit"])
            co2_flow = p["flow_low"] + fraction * (p["flow_boost"] - p["flow_low"])

        target_flow = max(co2_flow, base_flow)

        # Cold outdoor temp limiting
        if outdoor_temp <= p["outdoor_temp_low_limit"]:
            max_allowed = p["flow_low"]
        elif outdoor_temp >= p["outdoor_temp_high_limit"]:
            max_allowed = p["flow_max"]
        else:
            slope = (p["flow_max"] - p["flow_low"]) / (p["outdoor_temp_high_limit"] - p["outdoor_temp_low_limit"])
            max_allowed = p["flow_low"] + slope * (outdoor_temp - p["outdoor_temp_low_limit"])

        return min(target_flow, max_allowed)

    def return_air_compensation(self, return_air_temp):
        p = self.p
        if return_air_temp <= p["return_air_temp_low"]:
            return p["sup_temp_at_low"]
        if return_air_temp >= p["return_air_temp_high"]:
            return p["sup_temp_at_high"]
        slope = (p["sup_temp_at_high"] - p["sup_temp_at_low"]) / (p["return_air_temp_high"] - p["return_air_temp_low"])
        return float(np.clip(
            p["sup_temp_at_low"] + slope * (return_air_temp - p["return_air_temp_low"]),
            min(p["sup_temp_at_low"], p["sup_temp_at_high"]),
            max(p["sup_temp_at_low"], p["sup_temp_at_high"]),
        ))

    def _update_outdoor_avg(self, outdoor_temp):
        """Update 24h rolling average of outdoor temp (96 steps at 15min)."""
        self._outdoor_temps.append(outdoor_temp)
        if len(self._outdoor_temps) > 96:
            self._outdoor_temps = self._outdoor_temps[-96:]
        self._outdoor_24h_avg = sum(self._outdoor_temps) / len(self._outdoor_temps)

    def _s1_bands(self):
        """Compute Finnish S1 comfort bands from 24h outdoor average."""
        t = self._outdoor_24h_avg if self._outdoor_24h_avg is not None else 0.0
        if t <= 0:
            lower = 20.5
            upper = 22.0
        elif t <= 15:
            lower = 20.5 + 0.075 * t  # reaches 21.625 at t=15
            upper = 22.5 + 0.166 * t  # reaches 24.99 at t=15
        elif t <= 20:
            lower = 20.5 + 0.075 * t  # reaches 22.0 at t=20
            upper = 25.0
        else:
            lower = 22.0
            upper = 25.0
        return lower, upper

    def zone_setpoints_s1(self):
        """Compute setpoints that target the S1 band midpoint with margins."""
        p = self.p
        lower, upper = self._s1_bands()
        margin_htg = p.get("s1_htg_margin", 0.4)  # how far above S1 lower bound
        margin_clg = p.get("s1_clg_margin", 0.3)  # how far below S1 upper bound
        htg = lower + margin_htg
        clg = upper - margin_clg
        # Ensure reasonable deadband
        if clg - htg < 0.3:
            mid = (lower + upper) / 2.0
            htg = mid - 0.15
            clg = mid + 0.15
        return float(htg), float(clg)

    def calculate_setpoints(self, zone_temp, outdoor_temp, return_air_temp, occupancy,
                            hour, day, co2_concentration, direct_solar=0.0, wind_speed=0.0):
        p = self.p
        self._update_outdoor_avg(outdoor_temp)

        supply_air_temp = self.return_air_compensation(return_air_temp)
        htg, clg = self.zone_setpoints_s1()

        # Solar pre-cooling: lower cooling SP when strong sun to prevent overshoot
        if p["solar_precool_enabled"] and direct_solar > p["solar_precool_threshold"]:
            clg = max(18.0, clg + p["solar_precool_clg_offset"])

        # Wind chill compensation: raise heating SP in cold windy weather
        if p["wind_comp_enabled"] and wind_speed > p["wind_speed_threshold"] and outdoor_temp < p["wind_cold_outdoor_threshold"]:
            htg = min(25.0, htg + p["wind_htg_offset"])

        # Ensure htg <= clg
        if htg > clg:
            htg = clg - 0.3

        flow = self.co2_flow_control(hour, day, co2_concentration, outdoor_temp)
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
        wind_speed = self.get_variable("wind_speed", state)
        hour = float(self.api.exchange.hour(state))
        day = float(self.api.exchange.day_of_week(state))

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
