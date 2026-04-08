"""Compute total_cost_eur from an EnergyPlus eplusout.csv file.

Mirrors the exact hackathon scoring logic from visualize_output.ipynb.
"""

import pandas as pd
import numpy as np

COLUMN_NAMES = [
    'Time', 'Outdoor_Tdb_C', 'Outdoor_Twb_C',
    'Space1_occupants', 'Space2_occupants', 'Space3_occupants',
    'Space4_occupants', 'Space5_occupants',
    'lights-1', 'lights-2', 'lights-3', 'lights-4', 'lights-5',
    'equip-1', 'equip-2', 'equip-3', 'equip-4', 'equip-5',
    'Plenum1_T_C', 'Plenum1_RH_%',
    'Space1_T_C', 'Space1_RH_%',
    'Space2_T_C', 'Space2_RH_%',
    'Space3_T_C', 'Space3_RH_%',
    'Space4_T_C', 'Space4_RH_%',
    'Space5_T_C', 'Space5_RH_%',
    'Plenum_CO2_ppm', 'Plenum_CO2_pred', 'Plenum_CO2_setpoint_ppm', 'Plenum_CO2_internal_gain',
    'Space1_CO2_ppm', 'Space1_CO2_pred', 'Space1_CO2_setpoint_ppm', 'Space1_CO2_internal_gain',
    'Space2_CO2_ppm', 'Space2_CO2_pred', 'Space2_CO2_setpoint_ppm', 'Space2_CO2_internal_gain',
    'Space3_CO2_ppm', 'Space3_CO2_pred', 'Space3_CO2_setpoint_ppm', 'Space3_CO2_internal_gain',
    'Space4_CO2_ppm', 'Space4_CO2_pred', 'Space4_CO2_setpoint_ppm', 'Space4_CO2_internal_gain',
    'Space5_CO2_ppm', 'Space5_CO2_pred', 'Space5_CO2_setpoint_ppm', 'Space5_CO2_internal_gain',
    'doas_fan', 'fcu_1', 'fcu_2', 'fcu_3', 'fcu_4', 'fcu_5',
    'hex', 'chiller', 'tower', 'boiler',
    'coldw_pump', 'condw_pump', 'hotw_pump',
    'Node2_T_C', 'Node2_Mdot_kg/s', 'Node2_W_Ratio',
    'Node2_SP_T_C', 'Node2_CO2_ppm', 'Node1_T_C',
    'Gas_Facility_E_J', 'Elec_Facility_E_J', 'Elec_HVAC_E_J',
    'CoolingCoils:EnergyTransfer', 'HeatingCoils:EnergyTransfer',
    'ElectricityNet:Facility',
    'General:Cooling:EnergyTransfer', 'Cooling:EnergyTransfer',
]

ELEC_PRICE = 0.11
GAS_PRICE = 0.06
J_TO_KWH = 1 / 3.6e6
CLASS_THRESHOLD = 0.90

CO2_P1 = 2; CO2_P2 = 10; CO2_P3 = 50
TEMP_P1 = 1; TEMP_P2 = 5; TEMP_P3 = 25


def load_eplusout(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['Date/Time'] = df['Date/Time'].str.strip()
    mask_24 = df['Date/Time'].str.contains('24:00:00')
    df.loc[mask_24, 'Date/Time'] = df.loc[mask_24, 'Date/Time'].str.replace('24:00:00', '00:00:00')
    df['Date/Time'] = '2024/' + df['Date/Time']
    df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%Y/%m/%d %H:%M:%S')
    df.loc[mask_24, 'Date/Time'] += pd.Timedelta(days=1)
    df.columns = COLUMN_NAMES
    return df


def compute_total_cost(df: pd.DataFrame) -> dict:
    # Energy cost
    df_t = df.copy()
    df_t['Time'] = pd.to_datetime(df_t['Time'])
    monthly = df_t.groupby(df_t['Time'].dt.to_period('M')).agg(
        elec_kWh=('Elec_Facility_E_J', lambda x: x.sum() * J_TO_KWH),
        gas_kWh=('Gas_Facility_E_J', lambda x: x.sum() * J_TO_KWH),
    )
    energy_cost = (monthly['elec_kWh'] * ELEC_PRICE + monthly['gas_kWh'] * GAS_PRICE).sum()

    # CO2 penalty
    co2_cost = 0.0
    for i in range(1, 6):
        co2 = df[f'Space{i}_CO2_ppm']
        co2_cost += CO2_P1 * ((co2 > 770) & (co2 <= 970)).sum()
        co2_cost += CO2_P2 * ((co2 > 970) & (co2 <= 1220)).sum()
        co2_cost += CO2_P3 * (co2 > 1220).sum()

    # Temperature penalty
    t_out = df['Outdoor_Tdb_C'].rolling(24, min_periods=24).mean()
    t = t_out.to_numpy(dtype=float)
    lower_S1 = np.where(t <= 0, 20.5, np.where(t <= 20, 20.5 + 0.075 * t, 22.0))
    upper_S1 = np.where(t <= 0, 22.0, np.where(t <= 15, 22.5 + 0.166 * t, 25.0))
    lower_S2 = np.where(t <= 0, 20.5, np.where(t <= 20, 20.5 + 0.025 * t, 21.0))
    upper_S2 = np.where(t <= 0, 23.0, np.where(t <= 15, 23.0 + 0.20 * t, 26.0))
    lower_S3 = np.full_like(t, 20.0)
    upper_S3 = np.where(t <= 10, 25.0, 27.0)

    temp_cost = 0.0
    for i in range(1, 6):
        ti = pd.to_numeric(df[f'Space{i}_T_C'], errors='coerce').to_numpy(dtype=float)
        in_s1 = (ti >= lower_S1) & (ti <= upper_S1)
        in_s2 = (ti >= lower_S2) & (ti <= upper_S2)
        in_s3 = (ti >= lower_S3) & (ti <= upper_S3)
        temp_cost += TEMP_P1 * (~in_s1 & in_s2).sum()
        temp_cost += TEMP_P2 * (~in_s2 & in_s3).sum()
        temp_cost += TEMP_P3 * (~in_s3).sum()

    return {
        'energy_cost_eur': float(energy_cost),
        'co2_penalty_eur': float(co2_cost),
        'temp_penalty_eur': float(temp_cost),
        'total_cost_eur': float(energy_cost + co2_cost + temp_cost),
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "eplus_out/eplusout.csv"
    df = load_eplusout(path)
    costs = compute_total_cost(df)
    print(f"Energy: {costs['energy_cost_eur']:.2f} | CO2: {costs['co2_penalty_eur']:.2f} | Temp: {costs['temp_penalty_eur']:.2f} | TOTAL: {costs['total_cost_eur']:.2f}")
