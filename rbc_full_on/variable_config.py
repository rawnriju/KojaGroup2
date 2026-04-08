"""EnergyPlus variable, actuator, and meter handle configuration.

This file defines all the EnergyPlus I/O points used by the controller.
Each dictionary maps a short *alias* to the EnergyPlus identifiers needed
to obtain a runtime handle.

Mapping formats:
    VARIABLES : alias -> (variable_name, key_value)
    ACTUATORS : alias -> (component_type, control_type, actuator_key)
    METERS    : alias -> meter_name

The aliases are used throughout the controller code to read/write values
without referencing raw EnergyPlus strings everywhere.

Zone names correspond to the DOE reference small-office DOAS model:
    SPACE1-1 … SPACE5-1  (five conditioned zones)
    PLENUM-1             (return-air plenum)
"""

from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Output variables  (read-only sensor values)
# ---------------------------------------------------------------------------
#   alias -> (EnergyPlus variable name, key / zone name)

VARIABLES: Dict[str, Tuple[str, str]] = {
    # --- Site ---
    "outdoor_temp": (
        "Site Outdoor Air DryBulb Temperature",
        "Environment",
    ),

    # --- Plenum ---
    "plenum_temp": (
        "Zone Air Temperature",
        "PLENUM-1 THERMAL ZONE",
    ),
    "co2_plenum": (
        "Zone Air CO2 Concentration",
        "PLENUM-1 THERMAL ZONE",
    ),

    # --- Zone 1 ---
    "space1_temp":  ("Zone Air Temperature",        "SPACE1-1 THERMAL ZONE"),
    "space1_rh":    ("Zone Air Relative Humidity",   "SPACE1-1 THERMAL ZONE"),
    "space1_occupancy":  ("Zone People Occupant Count",   "SPACE1-1 THERMAL ZONE"),
    "space1_co2":   ("Zone Air CO2 Concentration",   "SPACE1-1 THERMAL ZONE"),

    # --- Zone 2 ---
    "space2_temp":  ("Zone Air Temperature",        "SPACE2-1 THERMAL ZONE"),
    "space2_rh":    ("Zone Air Relative Humidity",   "SPACE2-1 THERMAL ZONE"),
    "space2_occupancy":  ("Zone People Occupant Count",   "SPACE2-1 THERMAL ZONE"),
    "space2_co2":   ("Zone Air CO2 Concentration",   "SPACE2-1 THERMAL ZONE"),

    # --- Zone 3 ---
    "space3_temp":  ("Zone Air Temperature",        "SPACE3-1 THERMAL ZONE"),
    "space3_rh":    ("Zone Air Relative Humidity",   "SPACE3-1 THERMAL ZONE"),
    "space3_occupancy":  ("Zone People Occupant Count",   "SPACE3-1 THERMAL ZONE"),
    "space3_co2":   ("Zone Air CO2 Concentration",   "SPACE3-1 THERMAL ZONE"),

    # --- Zone 4 ---
    "space4_temp":  ("Zone Air Temperature",        "SPACE4-1 THERMAL ZONE"),
    "space4_rh":    ("Zone Air Relative Humidity",   "SPACE4-1 THERMAL ZONE"),
    "space4_occupancy":  ("Zone People Occupant Count",   "SPACE4-1 THERMAL ZONE"),
    "space4_co2":   ("Zone Air CO2 Concentration",   "SPACE4-1 THERMAL ZONE"),

    # --- Zone 5 ---
    "space5_temp":  ("Zone Air Temperature",        "SPACE5-1 THERMAL ZONE"),
    "space5_rh":    ("Zone Air Relative Humidity",   "SPACE5-1 THERMAL ZONE"),
    "space5_occupancy":  ("Zone People Occupant Count",   "SPACE5-1 THERMAL ZONE"),
    "space5_co2":   ("Zone Air CO2 Concentration",   "SPACE5-1 THERMAL ZONE"),
}


# ---------------------------------------------------------------------------
# Actuators  (write-back control points)
# ---------------------------------------------------------------------------
#   alias -> (component_type, control_type, actuator_key)

ACTUATORS: Dict[str, Tuple[str, str, str]] = {
    "clg_setpoint": (
        "Schedule:Compact", "Schedule Value", "CLG-SETP-SCH",
    ),
    "htg_setpoint": (
        "Schedule:Compact", "Schedule Value", "HTG-SETP-SCH",
    ),
    "ahu_temperature_setpoint": (
        "Schedule:Compact", "Schedule Value", "AHU_Supply_Temp_Schedule",
    ),
    "ahu_mass_flow_rate_setpoint": (
        "Fan", "Fan Air Mass Flow Rate", "DOAS SYSTEM SUPPLY FAN",
    ),
}


# ---------------------------------------------------------------------------
# Meters  (cumulative energy / resource tracking)
# ---------------------------------------------------------------------------
#   alias -> EnergyPlus meter name

METERS: Dict[str, str] = {
    "electricity_hvac":             "Electricity:HVAC",
    "gas_total":                    "NaturalGas:Facility",
    "coolingcoil_energytransfer":   "CoolingCoils:EnergyTransfer",
    "heatingcoil_energytransfer":   "HeatingCoils:EnergyTransfer",
    "net_elec":                     "ElectricityNet:Facility",
    "fans_electricity":             "Fans:Electricity",
}
