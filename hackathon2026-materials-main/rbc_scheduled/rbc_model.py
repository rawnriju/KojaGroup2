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
import numpy as np

# Constants

# Zone heating setpoint [°C] — linear ramp between outdoor temp bounds
ZONE_HTG_SETPOINT_LOW  = 21.0   # heating setpoint at outdoor <= 0 °C
ZONE_HTG_SETPOINT_HIGH = 22.0   # heating setpoint at outdoor >= 20 °C

# Zone cooling setpoint [°C] — linear ramp between outdoor temp bounds
ZONE_CLG_SETPOINT_LOW  = 23.0   # cooling setpoint at outdoor <= 0 °C
ZONE_CLG_SETPOINT_HIGH = 25.0   # cooling setpoint at outdoor >= 20 °C

OUTDOOR_TEMP_LOW = 0.0          # outdoor temp lower bound for compensation
OUTDOOR_TEMP_HIGH = 20.0        # outdoor temp upper bound for compensation

# Supply air temperature compensation curve (linear between two points)
RETURN_AIR_TEMP_LOW = 21.5      # return air temp at which supply = SUP_TEMP_AT_LOW
RETURN_AIR_TEMP_HIGH = 24.5     # return air temp at which supply = SUP_TEMP_AT_HIGH
SUP_TEMP_AT_LOW = 19.0          # supply air temp when return air is cold
SUP_TEMP_AT_HIGH = 17.0         # supply air temp when return air is warm

# CO2 demand-controlled ventilation thresholds [ppm]
CO2_MIN_LIMIT = 500
CO2_MAX_LIMIT = 750

OUTDOOR_TEMP_LOW_LIMIT = -25.0          # outdoor temp lower bound for compensation
OUTDOOR_TEMP_HIGH_LIMIT = -15.0        # outdoor temp upper bound for compensation

# AHU fan mass flow rates [kg/s]
FLOW_LOW = 0.20        # weekend / holiday
FLOW_MODERATE = 0.60   # workday, outside working hours
FLOW_BOOST = 1.0      # workday during working hours, or CO2 boost
FLOW_MAX = 1.0        # physical upper limit



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
        self.htg_setpoint: float = ZONE_HTG_SETPOINT_LOW
        self.clg_setpoint: float = ZONE_CLG_SETPOINT_LOW

        # CO2 demand-boosting state
        self._co2_boosting: bool = False

        # Named operating modes (for logging / reference)
        self.mode_params = {
            0: AHUModeParams(flow=0.0, mode_name="off"),
            1: AHUModeParams(flow=FLOW_LOW, mode_name="low"),
            2: AHUModeParams(flow=FLOW_MODERATE, mode_name="moderate"),
            3: AHUModeParams(flow=FLOW_BOOST, mode_name="boost"),
        }

    
    # Flow control

    def co2_flow_control(self, hour: float, day: int, co2_concentration: float, outdoortemp: float) -> float:
        """Return AHU fan mass flow rate based on time-of-day schedule and CO2 concentration.

        Args:
            hour: Current simulation hour (fractional, 0–24).
            day:  Day of the week (1 = Sunday … 7 = Saturday).
            co2_concentration: Max zone CO2 concentration [ppm].


        Returns:
            Mass flow rate [kg/s].
        """
        is_workday = day != 1 and day != 7
        is_nightflush = is_workday and 1.5 <= hour < 2.5 # mode 1
        is_nightflush_weekend = not is_workday and 0.0 <= hour < 2.0 # mode 1
        is_before_work_flush = is_workday and 4.5 <= hour < 5.5 # mode 2
        is_weekend_flush = not is_workday and 6.0 <= hour < 17.0 # mode 1
        is_working_hours = is_workday and 5.5 <= hour < 23.5 # mode 1

        # Default schedule-based flow control
        # Workday: 1.30-2.30 low, 4.30-5.30 moderate, 5.30-23.30 low, otherwise off
        # Weekend: 0.00-2.00 low, 6.00-17.00 low, otherwise off

        base_flow = 0.0
        if is_before_work_flush:
            base_flow = FLOW_MODERATE
        elif is_nightflush or is_nightflush_weekend or is_weekend_flush or is_working_hours:
            base_flow = FLOW_LOW
        else:
            base_flow = 0.0


        # CO2
        co2_flow = 0.0
        # Työpäivän aikana CO2-pohjainen säätö
        if co2_concentration <= CO2_MIN_LIMIT:
            co2_flow = 0.0
        elif co2_concentration >= CO2_MAX_LIMIT:
            co2_flow = FLOW_BOOST
        else:
            # Lineaarinen interpolointi välillä 500-750 ppm
            fraction = (co2_concentration - CO2_MIN_LIMIT) / (CO2_MAX_LIMIT - CO2_MIN_LIMIT)
            co2_flow = FLOW_LOW + fraction * (FLOW_BOOST - FLOW_LOW)


        target_flow = max(co2_flow, base_flow)  # CO2 boost can only increase flow, never decrease it

        # Outdoor temperature limit
        max_allowed = target_flow  # default to target flow if no limits apply
        if outdoortemp <= OUTDOOR_TEMP_LOW_LIMIT:
            max_allowed = FLOW_LOW
        elif outdoortemp >= OUTDOOR_TEMP_HIGH_LIMIT:
            max_allowed = FLOW_MAX
        else:
            slope = (FLOW_MAX - FLOW_LOW) / (OUTDOOR_TEMP_HIGH_LIMIT - OUTDOOR_TEMP_LOW_LIMIT)
            max_allowed = FLOW_LOW + slope * (outdoortemp - OUTDOOR_TEMP_LOW_LIMIT)

        return min(target_flow, max_allowed)

    # Supply air temperature compensation
    

    def return_air_compensation(self, return_air_temp: float) -> float:
        """Compute supply air temperature setpoint from return air temperature.

        Uses a linear compensation curve between two reference points:
        - return_air <= 21.5 °C  →  supply = 19.0 °C
        - return_air >= 24.5 °C  →  supply = 17.0 °C

        Args:
            return_air_temp: Measured return air temperature [°C].

        Returns:
            Supply air temperature setpoint [°C].
        """
        if return_air_temp <= RETURN_AIR_TEMP_LOW:
            return SUP_TEMP_AT_LOW
        if return_air_temp >= RETURN_AIR_TEMP_HIGH:
            return SUP_TEMP_AT_HIGH

        # Linear interpolation between the two reference points
        slope = (SUP_TEMP_AT_HIGH - SUP_TEMP_AT_LOW) / (RETURN_AIR_TEMP_HIGH - RETURN_AIR_TEMP_LOW)
        return np.clip(SUP_TEMP_AT_LOW + slope * (return_air_temp - RETURN_AIR_TEMP_LOW), SUP_TEMP_AT_LOW, SUP_TEMP_AT_HIGH)


    def zone_setpoints(self, outdoor_temp: float) -> Tuple[float, float]:
        """Compute zone heating/cooling setpoints from outdoor temperature.

        Both setpoints are linearly interpolated between their low and high
        values over the outdoor temperature range [OUTDOOR_TEMP_LOW, OUTDOOR_TEMP_HIGH].
        """
        t = np.clip((outdoor_temp - OUTDOOR_TEMP_LOW) / (OUTDOOR_TEMP_HIGH - OUTDOOR_TEMP_LOW), 0.0, 1.0)
        htg = ZONE_HTG_SETPOINT_LOW + t * (ZONE_HTG_SETPOINT_HIGH - ZONE_HTG_SETPOINT_LOW)
        clg = ZONE_CLG_SETPOINT_LOW + t * (ZONE_CLG_SETPOINT_HIGH - ZONE_CLG_SETPOINT_LOW)
        return float(htg), float(clg)
    
    
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
            hour:               Current simulation hour (0–24).
            day:                Day of the week (1 = Mon … 7 = Sun).
            co2_concentration:  Maximum zone CO2 concentration [ppm].

        Returns:
            Tuple of:
                - heating_setpoint [°C]
                - cooling_setpoint [°C]
                - supply_air_temp  [°C]
                - fan_flow_rate    [kg/s]
        """

        # --- Supply air temperature (return-air-compensated) ---
        supply_air_temp = self.return_air_compensation(return_air_temp)

        # --- Zone setpoints (outdoor-temperature-compensated, with deadband) ---
        htg, clg = self.zone_setpoints(outdoor_temp)

        self.htg_setpoint = htg
        self.clg_setpoint = clg

        # --- Fan flow rate (schedule + CO2 boost override) ---
        flow = self.co2_flow_control(hour, day, co2_concentration, outdoor_temp)

        return self.htg_setpoint, self.clg_setpoint, supply_air_temp, flow


