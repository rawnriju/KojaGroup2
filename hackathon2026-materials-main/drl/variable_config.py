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

"""EnergyPlus variable, actuator, and meter handle configuration.

This file defines all the EnergyPlus I/O points used by the controller.
Each dictionary maps a short *alias* to the EnergyPlus identifiers needed
to obtain a runtime handle.

Mapping formats:
    SENSORS : alias -> (variable_name, key_value)
    ACTUATORS : alias -> (component_type, control_type, actuator_key)
    METERS    : alias -> meter_name

The aliases are used throughout the controller code to read/write values
without referencing raw EnergyPlus strings everywhere.

Zone names correspond to the DOE reference small-office DOAS model:
    SPACE1-1 … SPACE5-1  (five conditioned zones)
    PLENUM-1             (return-air plenum)
"""

from typing import Dict, Tuple, List

# Possible output variables (observations/states) and actuators (actions)

ZONE_NAMES = [
    "SPACE1-1 THERMAL ZONE",
    "SPACE2-1 THERMAL ZONE",
    "SPACE3-1 THERMAL ZONE",
    "SPACE4-1 THERMAL ZONE",
    "SPACE5-1 THERMAL ZONE",
]

# alias, variable_name, key_value, request_flag
SENSOR_DEF: List[Tuple[str, str, str, bool]] = [
    ("plenum_temp", "Zone Air Temperature", "PLENUM-1 THERMAL ZONE", False),
    ("plenum_co2", "Zone Air CO2 Concentration", "PLENUM-1 THERMAL ZONE", False),
    ("outdoor_temp", "Site Outdoor Air Drybulb Temperature", "Environment", False),
]
# + solar radiation / outdoor rh / wind speed / etc. ?
# return air temp eli plenum? vai
for i, zone in enumerate(ZONE_NAMES, start=1):
    SENSOR_DEF += [
        (f"space{i}_temp", "Zone Air Temperature", zone, False),
        (f"space{i}_rh", "Zone Air Relative Humidity", zone, False),
        (f"space{i}_occ", "Zone People Occupant Count", zone, False),
        (f"space{i}_co2", "Zone Air CO2 Concentration", zone, False),
    ]

# alias, component_type, control_type, key_value
ACTUATOR_DEF: List[Tuple[str, str, str, str]] = [
    ("heating_setpoint", "Schedule:Compact", "Schedule Value", "HTG-SETP-SCH"),
    ("cooling_setpoint", "Schedule:Compact", "Schedule Value", "CLG-SETP-SCH"),
    ("ahu_supply_temp", "Schedule:Compact", "Schedule Value", "AHU_Supply_Temp_Schedule"),
    ("supply_fan_flow", "Fan", "Fan Air Mass Flow Rate", "DOAS SYSTEM SUPPLY FAN"),
]

# alias, meter_name
METER_DEF: List[Tuple[str, str]] = [
    ("electricity_hvac", "Electricity:HVAC"), 
    ("fans_electricity", "Fans:Electricity"),
    ("cooling_energy", "CoolingCoils:EnergyTransfer"),
    ("heating_energy", "HeatingCoils:EnergyTransfer"),
    ("facility_electricity_net", "ElectricityNet:Facility"),
    ("gas_total", "NaturalGas:Facility"),
]
