"""Rule-Based Control (RBC) model for AHU and zone setpoint management.

This module implements a simple rule-based controller that determines:
- Zone heating and cooling temperature setpoints (outdoor-compensated)
- Supply air temperature setpoint (return-air-compensated)
- AHU fan mass flow rate (schedule-based with CO2 demand boosting)

The control logic follows typical HVAC best practices for a DOAS
(Dedicated Outdoor Air System) with fan coil units.
"""

from dataclasses import dataclass
from typing import Tuple

# Constants

# Zone temperature setpoint range [°C]
ZONE_TEMP_SETPOINT = 22.5   # setpoint

# Supply air temperature setpoint range [°C] 
SUPPLY_AIR_TEMP_SETPOINT = 19.0   # setpoint

# AHU fan mass flow rates [kg/s]
FLOW_LOW = 0.20  
FLOW_MODERATE = 0.60
FLOW_BOOST = 1.0
FLOW_MAX = 1.0


@dataclass
class AHUModeParams:
    """Parameters for a single AHU operating mode."""
    flow: float
    mode_name: str


class RBCModel:
    """Rule-based HVAC controller for a 5-zone DOAS building.

    The model computes heating/cooling setpoints, supply air temperature,
    and fan flow rate at every simulation timestep.  It is intended to be
    called from an EnergyPlus runtime callback.
    """

    def __init__(self) -> None:
        # Current zone setpoints (updated each timestep)
        self.htg_setpoint: float = ZONE_TEMP_SETPOINT
        self.clg_setpoint: float = ZONE_TEMP_SETPOINT

        # CO2 demand-boosting state
        self._co2_boosting: bool = False
    
    
    # Main setpoint calculation 

    def calculate_setpoints(
        self,
        zone_temp: float,
        outdoor_temp: float,
        return_air_temp: float,
        occupancy: float,
        hour: float,
        day: int,
        co2_concentration: float,
    ) -> Tuple[float, float, float, float]:
        """Calculate all control setpoints for the current timestep.

        Args:
            zone_temp:          Representative zone air temperature [°C].
            outdoor_temp:       Outdoor dry-bulb temperature [°C].
            return_air_temp:    AHU return air temperature [°C].
            occupancy:          Current occupancy count (reserved for future use).
            hour:               Current simulation hour (0-24).
            day:                Day of the week (1 = Mon … 7 = Sun).
            co2_concentration:  Maximum zone CO2 concentration [ppm].

        Returns:
            Tuple of:
                - heating_setpoint [°C]
                - cooling_setpoint [°C]
                - supply_air_temp  [°C]
                - fan_flow_rate    [kg/s]
        """

        # --- Supply air temperature (19°C all the time) ---
        supply_air_temp = SUPPLY_AIR_TEMP_SETPOINT

        # --- Zone setpoints (22.5°C all the time) ---

        self.htg_setpoint = ZONE_TEMP_SETPOINT
        self.clg_setpoint = ZONE_TEMP_SETPOINT

        # --- Fan flow rate (full speed all time) ---
        flow = FLOW_MAX

        return self.htg_setpoint, self.clg_setpoint, supply_air_temp, flow


