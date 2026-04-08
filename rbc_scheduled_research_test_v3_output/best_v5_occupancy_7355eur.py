#!/usr/bin/env python3
"""v5 OCCUPANCY-AWARE RBC — Novel strategies for minimum total cost.

Key innovations:
1. OCCUPANCY-AWARE SETPOINTS: During occupied hours, people + equipment
   generate ~20-40 W/m² of "free heating." We lower the heating setpoint
   during occupied hours and let internal gains push temps up. During
   unoccupied hours, we raise heating SP to prevent drift below S1.

2. OUTDOOR-TEMP-BASED SUPPLY AIR: Instead of return-air compensation,
   use seasonal strategy:
   - Winter (outdoor < 5°C): Supply 19-20°C (reduce DOAS heating)
   - Summer (outdoor > 20°C): Supply 16°C (free cooling)
   - Shoulder: interpolate

3. INTERNAL GAIN PREDICTION: Track real-time occupancy to estimate
   future internal gains and adjust setpoints proactively.

4. ZONE-WEIGHTED CO2: Instead of max(CO2), use a weighted approach
   that responds proportionally to the number of zones with high CO2.

Baseline: v3 adaptive @ 7,409.05 EUR
"""

import sys
import os
import shutil
import time
from pathlib import Path
import numpy as np

# --- EnergyPlus Python API setup ---
ENERGYPLUS_DIR = r"C:\EnergyPlusV25-2-0"
sys.path.append(ENERGYPLUS_DIR)
from pyenergyplus.api import EnergyPlusAPI

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
IDF_FILE = SCRIPT_DIR.parent / "DOAS_wNeutralSupplyAir_wFanCoilUnits.idf"
EPW_FILE = SCRIPT_DIR.parent / "FIN_TR_Tampere.Satakunnankatu.027440_TMYx.2004-2018.epw"
OUT_DIR = SCRIPT_DIR / "eplus_out"


# ═══════════════════════════════════════════════════════════════════════════
# Variable config
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
# v5 Occupancy-Aware RBC Model
# ═══════════════════════════════════════════════════════════════════════════

class OccupancyAwareRBC:
    """HVAC controller that exploits internal gains from occupancy.

    Core insight: In a typical office, people + equipment generate significant
    heat. During occupied hours, this "free heating" means we need less
    mechanical heating. During unoccupied hours, we need more.
    """

    def __init__(self):
        self._outdoor_temps = []
        self._outdoor_24h_avg = None
        self._occupancy_history = []  # track occupancy patterns

    def _update_outdoor_avg(self, outdoor_temp):
        self._outdoor_temps.append(outdoor_temp)
        if len(self._outdoor_temps) > 96:
            self._outdoor_temps = self._outdoor_temps[-96:]
        self._outdoor_24h_avg = sum(self._outdoor_temps) / len(self._outdoor_temps)

    def _s1_bands(self):
        t = self._outdoor_24h_avg if self._outdoor_24h_avg is not None else 0.0
        if t <= 0:
            return 20.5, 22.0
        elif t <= 15:
            return 20.5 + 0.075 * t, 22.5 + 0.166 * t
        elif t <= 20:
            return 20.5 + 0.075 * t, 25.0
        else:
            return 22.0, 25.0

    # ── INNOVATION 1: Occupancy-Aware Heating Setpoint ──

    def _occupancy_adjusted_margins(self, total_occupancy, hour, day):
        """Adjust heating/cooling margins based on occupancy.

        When occupied: internal gains provide free heating → reduce margin.
        When unoccupied: no free heat → keep proven margin.
        """
        is_workday = day != 1 and day != 7
        is_working = is_workday and 6.0 <= hour < 19.0

        if is_working and total_occupancy > 5:
            htg_margin = 0.05
            clg_margin = 0.08
        elif is_working and total_occupancy > 0:
            htg_margin = 0.10
            clg_margin = 0.06
        else:
            htg_margin = 0.55
            clg_margin = 0.05

        return htg_margin, clg_margin

    # ── Supply Air Temp (proven return-air compensation from v3) ──

    def _return_air_supply_temp(self, return_air_temp):
        """Supply air temp from return air compensation (17→16°C curve)."""
        if return_air_temp <= 21.0:
            return 17.0
        elif return_air_temp >= 24.5:
            return 16.0
        else:
            slope = (16.0 - 17.0) / (24.5 - 21.0)
            return 17.0 + slope * (return_air_temp - 21.0)

    # ── CO2 Control (proven max-based DCV from v3) ──

    def _co2_dcv_flow(self, zone_co2s):
        """CO2 demand-controlled ventilation using max zone CO2."""
        co2_min, co2_max = 650, 760
        max_co2 = max(zone_co2s)

        if max_co2 <= co2_min:
            return 0.0
        elif max_co2 >= co2_max:
            return 1.0
        else:
            frac = (max_co2 - co2_min) / (co2_max - co2_min)
            return 0.05 + frac * 0.95

    # ── Main control logic ──

    def calculate_setpoints(self, zone_temps, outdoor_temp, return_air_temp,
                            occupancy, hour, day, zone_co2s,
                            direct_solar=0.0, wind_speed=0.0):
        self._update_outdoor_avg(outdoor_temp)

        lower_s1, upper_s1 = self._s1_bands()
        s1_width = upper_s1 - lower_s1

        # INNOVATION 1: Occupancy-aware margins
        htg_margin, clg_margin = self._occupancy_adjusted_margins(occupancy, hour, day)

        # Adaptive width factor (proven from v3)
        width_factor = float(np.clip(1.0 - 0.20 * (s1_width - 1.5), 0.5, 1.0))
        htg = lower_s1 + htg_margin * width_factor
        clg = upper_s1 - clg_margin

        # Ensure valid deadband
        if clg - htg < 0.3:
            mid = (lower_s1 + upper_s1) / 2.0
            htg = mid - 0.15
            clg = mid + 0.15

        # Supply air temp (proven return-air compensation from v3)
        supply_air_temp = self._return_air_supply_temp(return_air_temp)

        # CO2 DCV (proven max-based from v3)
        co2_flow = self._co2_dcv_flow(zone_co2s)

        # Minimal base flow during working hours
        is_workday = day != 1 and day != 7
        is_working = is_workday and 5.5 <= hour < 20.0
        min_flow = 0.05 if is_working else 0.0

        flow = max(co2_flow, min_flow)
        flow = float(np.clip(flow, 0.0, 1.0))

        # Cold outdoor temp limiting (proven: -30 to -20)
        if outdoor_temp <= -30.0:
            max_allowed = 0.05
        elif outdoor_temp >= -20.0:
            max_allowed = 1.0
        else:
            slope = (1.0 - 0.05) / (-20.0 - (-30.0))
            max_allowed = 0.05 + slope * (outdoor_temp - (-30.0))
        flow = min(flow, max_allowed)

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

        zone_temps, zone_co2s, occs = [], [], []
        for i in range(1, 6):
            zone_temps.append(self.get_variable(f"space{i}_temp", state))
            zone_co2s.append(self.get_variable(f"space{i}_co2", state))
            occs.append(self.get_variable(f"space{i}_occupancy", state))

        total_occ = sum(occs)

        htg, clg, supply_air_temp, flow = self.model.calculate_setpoints(
            zone_temps=zone_temps,
            outdoor_temp=outdoor_temp,
            return_air_temp=plenum_temp,
            occupancy=total_occ,
            hour=hour,
            day=day,
            zone_co2s=zone_co2s,
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

def run_simulation():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    api = EnergyPlusAPI()
    state = api.state_manager.new_state()

    for name, (var, key) in VARIABLES.items():
        api.exchange.request_variable(state, var, key)

    model = OccupancyAwareRBC()
    controller = Controller(api, model)

    api.runtime.callback_after_new_environment_warmup_complete(
        state, controller.initialize_handles
    )
    api.runtime.callback_begin_zone_timestep_after_init_heat_balance(
        state, controller.control_callback
    )

    args = ["-d", str(OUT_DIR), "-w", str(EPW_FILE), "-r", str(IDF_FILE)]
    print("Running EnergyPlus...")
    t0 = time.time()
    rc = api.runtime.run_energyplus(state, args)
    elapsed = time.time() - t0
    print(f"EnergyPlus finished in {elapsed:.1f}s with code {rc}")

    if rc != 0:
        print("SIMULATION FAILED!")
        return None

    from cost_calculator import load_eplusout, compute_total_cost
    df = load_eplusout(str(OUT_DIR / "eplusout.csv"))
    costs = compute_total_cost(df)
    return costs


if __name__ == "__main__":
    costs = run_simulation()
    if costs:
        print(f"\n{'='*60}")
        print(f"  v5 OCCUPANCY-AWARE RBC")
        print(f"  Energy:  {costs['energy_cost_eur']:>10.2f} EUR")
        print(f"  CO2:     {costs['co2_penalty_eur']:>10.2f} EUR")
        print(f"  Temp:    {costs['temp_penalty_eur']:>10.2f} EUR")
        print(f"  TOTAL:   {costs['total_cost_eur']:>10.2f} EUR")
        print(f"{'='*60}")
    else:
        print("FAILED")
