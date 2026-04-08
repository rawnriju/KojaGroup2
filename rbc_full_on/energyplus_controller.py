"""EnergyPlus runtime controller for the Rule-Based Control (RBC) strategy.

This module bridges the EnergyPlus Python API with the RBC model.  It:
1. Initialises variable, actuator, and meter handles after warm-up.
2. Reads sensor values (zone temps, CO2, outdoor conditions) each timestep.
3. Passes them to the RBC model and writes the resulting setpoints back
   to EnergyPlus via actuator handles.

Usage:
    Instantiate ``EnergyPlusController`` with the API object and an
    ``RBCModel``, then register its methods as EnergyPlus callbacks
    (see ``run_idf.py``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List
import json

from variable_config import METERS, VARIABLES, ACTUATORS


class EnergyPlusController:
    """Reads sensor data from EnergyPlus and applies control setpoints.

    Args:
        api:   EnergyPlus Python API instance.
        model: Control-strategy object that exposes ``calculate_setpoints()``.
    """

    def __init__(self, api: Any, model: Any) -> None:
        self.api = api
        self.model = model
        self.handles: Dict[str, int] = {}


    # Handle initialisation


    def initialize_handles(self, state: Any) -> None:
        """Request and cache all variable, actuator, and meter handles.

        Must be called after warm-up is complete so that EnergyPlus
        has created the objects referenced in ``variable_config.py``.
        """
        ex = self.api.exchange

        for name, (var, key) in VARIABLES.items():
            self.handles[name] = ex.get_variable_handle(state, var, key)

        for name, (ctype, control, key) in ACTUATORS.items():
            self.handles[name] = ex.get_actuator_handle(state, ctype, control, key)

        for name, meter_name in METERS.items():
            self.handles[name] = ex.get_meter_handle(state, meter_name)

        # Warn about any handles that failed to resolve
        bad = [n for n, h in self.handles.items() if h == -1]
        if bad:
            print(f"WARNING: the following handles resolved to -1: {bad}")
        else:
            print(f"All {len(self.handles)} handles initialised successfully.")

    
    # Convenience getters / setters
    

    def get_variable(self, name: str, state: Any, default: float = 0.0) -> float:
        """Read a variable value from EnergyPlus.

        Args:
            name:    Key defined in ``VARIABLES`` (e.g. ``"zone1_temp"``).
            state:   Current EnergyPlus state pointer.
            default: Fallback value if the handle is missing.

        Returns:
            The current simulation value, or default.
        """
        handle = self.handles.get(name)
        if handle is None or handle == -1:
            return default
        if name in METERS:
            return self.api.exchange.get_meter_value(state, handle)
        return self.api.exchange.get_variable_value(state, handle)

    def set_actuator(self, name: str, value: float, state: Any) -> None:
        """Write a value to an EnergyPlus actuator.

        Args:
            name:  Key defined in ``ACTUATORS`` (e.g. ``"htg_setpoint"``).
            value: Desired actuator value.
            state: Current EnergyPlus state pointer.
        """
        handle = self.handles.get(name)
        if handle is not None and handle != -1:
            self.api.exchange.set_actuator_value(state, handle, value)

    def _collect_obs(self, state: Any) -> Dict:
        obs = {
            "outdoor_temp": self.get_variable("outdoor_temp", state),

            "plenum_temp": self.get_variable("plenum_temp", state),

            "space1_temp": self.get_variable("space1_temp", state),
            "space1_rh": self.get_variable("space1_rh", state),
            "space1_co2": self.get_variable("space1_co2", state),

            "space2_temp": self.get_variable("space2_temp", state),
            "space2_rh": self.get_variable("space2_rh", state),
            "space2_co2": self.get_variable("space2_co2", state),

            "space3_temp": self.get_variable("space3_temp", state),
            "space3_rh": self.get_variable("space3_rh", state),
            "space3_co2": self.get_variable("space3_co2", state),

            "space4_temp": self.get_variable("space4_temp", state),
            "space4_rh": self.get_variable("space4_rh", state),
            "space4_co2": self.get_variable("space4_co2", state),

            "space5_temp": self.get_variable("space5_temp", state),
            "space5_rh": self.get_variable("space5_rh", state),
            "space5_co2": self.get_variable("space5_co2", state),

            "electricity_hvac": self.get_variable("electricity_hvac", state),
            'gas_total': self.get_variable("gas_total", state),

            "hour": float(self.api.exchange.hour(state)),
            "day_of_week": float(self.api.exchange.day_of_week(state)),  
        }

        return obs
    


    # Main control callback
    

    def control_callback(self, state: Any) -> None:
        """Timestep callback: read sensors, compute setpoints, apply them.

        Registered with ``callback_begin_zone_timestep_after_init_heat_balance``.
        """
        if not self.handles:
            return  # handles not ready yet

        obs = self._collect_obs(state)

        space1_temp = obs.get("space1_temp", 22.0)
        space1_co2 = obs.get("space1_co2", 400.0)

        space2_temp = obs.get("space2_temp", 22.0)
        space2_co2 = obs.get("space2_co2", 400.0)   

        space3_temp = obs.get("space3_temp", 22.0)
        space3_co2 = obs.get("space3_co2", 400.0)

        space4_temp = obs.get("space4_temp", 22.0)
        space4_co2 = obs.get("space4_co2", 400.0)

        space5_temp = obs.get("space5_temp", 22.0)  
        space5_co2 = obs.get("space5_co2", 400.0)

        outdoor_temp = obs.get("outdoor_temp", 0.0)
        plenum_temp = obs.get("plenum_temp", 0.0)

        hour = obs.get("hour", 0.0)
        day_of_week = obs.get("day_of_week", 1.0)

        temps = [space1_temp, space2_temp, space3_temp, space4_temp, space5_temp]
        avg_temp = sum(temps) / len(temps)

        co2s = [space1_co2, space2_co2, space3_co2, space4_co2, space5_co2]
        max_co2 = max(co2s)

        # --- expert action ---
        htg, clg, supply_air_temp, flow = self.model.calculate_setpoints(
            zone_temp=avg_temp,
            outdoor_temp=outdoor_temp,
            return_air_temp=plenum_temp,
            occupancy=0,
            hour=hour,
            day=day_of_week,
            co2_concentration=max_co2,
        )

        # --- apply action ---
        self.set_actuator("htg_setpoint", htg, state)
        self.set_actuator("clg_setpoint", clg, state)
        self.set_actuator("ahu_temperature_setpoint", supply_air_temp, state)
        self.set_actuator("ahu_mass_flow_rate_setpoint", flow, state)

    
    # Debug / introspection
    

    def get_api_data(self, state: Any) -> None:
        """Dump all available EnergyPlus API data points to a text file.

        Useful for discovering variable names and keys during development.
        The output is written to ``api_initialization_log.txt`` in the
        current working directory.
        """
        api_data = self.api.exchange.get_api_data(state)
        with open("api_initialization_log.txt", "w", encoding="utf-8") as f:
            for item in api_data:
                f.write(
                    f"{item.name}, key: {item.key}, type: {item.type}, "
                    f"{item.unit}, {item.what}\n"
                )
        print(f"API data ({len(api_data)} items) written to api_initialization_log.txt")
