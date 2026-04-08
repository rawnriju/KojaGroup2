# HVAC Control Hackathon - Project Overview

## 🎯 Objective
The goal of this hackathon is to design an optimal HVAC (Heating, Ventilation, and Air Conditioning) controller for a 5-zone small office building. The building model is simulated over a full calendar year using **EnergyPlus** at 15-minute timesteps. It features a Dedicated Outdoor Air System (DOAS) with local Fan Coil Units (FCU).

The winning controller will **minimize the Total Cost**, which is a balance of three competing objectives:

1. **Energy Efficiency**: Minimizing electricity (0.11 €/kWh) and natural gas (0.06 €/kWh) consumption.
2. **Air Quality (CO2 Penalty)**: Keeping CO2 concentrations below specific thresholds. Penalties apply independently to each of the 5 zones for hours above thresholds (using a 90% compliance rule):
   - > 770 ppm to 970 ppm: 2 €/h
   - > 970 ppm to 1220 ppm: 10 €/h
   - > 1220 ppm: 50 €/h
3. **Thermal Comfort (Temperature Penalty)**: Keeping zone temperatures inside comfort bands (Finnish S1/S2/S3 classes) which dynamically adjust based on a 24-hour rolling average of the outdoor temperature. Penalties apply for hours outside the bands:
   - Outside S1 but inside S2: 1 €/h
   - Outside S2 but inside S3: 5 €/h
   - Outside S3: 25 €/h

At each timestep, the controller receives sensor readings (temperatures, CO2, occupancy, time, etc.) and must output 4 actions:
- `heating_setpoint` (18.0 – 25.0 °C)
- `cooling_setpoint` (18.0 – 25.0 °C)
- `ahu_supply_temp` (16.0 – 21.0 °C)
- `supply_fan_flow` (0.0 – 1.0 kg/s)

---

## 🛠️ The Three Approaches
The repository provides three starting templates to tackle the problem:

### 1. `rbc_full_on` (The Baseline)
A basic rule-based controller (RBC) that operates at maximum capacity.
- **How it works:** Hardcodes setpoints to 22.5 °C for both heating and cooling, and runs the fan constantly at maximum flow (1.0 kg/s).
- **Outcome:** Excellent air quality and temperature comfort, but incurs massive energy costs. This is the baseline to beat.

### 2. `rbc_scheduled` (The Smart Rules)
A smarter rule-based controller utilizing heuristics and `if/else` logic.
- **How it works:** Adjusts the supply air temperature based on the return air, shifts zone temperature setpoints dynamically based on the outdoor temperature to stay within the S1 band, and uses Demand-Controlled Ventilation (DCV) to linearly ramp up the fan only when CO2 levels rise.
- **Outcome:** Significantly lower energy consumption than the baseline, though reliant on human-crafted rules that may not find the absolute mathematical optimum.

### 3. `drl` (Deep Reinforcement Learning)
A machine learning approach using the Soft Actor-Critic (SAC) algorithm from `Stable-Baselines3`.
- **How it works:** Wraps the EnergyPlus simulation in a Gymnasium environment. The agent learns the optimal policy through trial and error over millions of timesteps.
- **Outcome:** Has the highest potential for finding complex, mathematically optimal control policies (like pre-cooling or pre-heating based on anticipated occupancy), provided the reward function and observation space are designed correctly.

---

## 📁 Codebase Deep Dive

The repository is structured to keep the three approaches isolated but sharing the same core building model (`.idf`) and weather file (`.epw`).

### 1. The Rule-Based Controllers (`rbc_full_on/` & `rbc_scheduled/`)
Both RBC folders share a similar architecture:
- **`run_idf.py`**: The entry point. It sets up paths, instantiates the EnergyPlus API, registers the callbacks for the controller, and runs the simulation. It also handles saving the trajectories to `expert_data.json` which can be used for behavioral cloning in the DRL approach.
- **`rbc_model.py` / `rbc_model_1.py`**: This is where the core strategy lives. It implements `calculate_setpoints()` (or specific methods in a class), taking sensor readings as inputs and returning the 4 control values. **This is the main file you edit for RBC.**
- **`energyplus_controller.py`**: The bridge between the RBC model and the EnergyPlus Python API. It requests runtime handles during warm-up and triggers the control logic callback every timestep to read sensors and apply the model's chosen setpoints.
- **`variable_config.py`**: Maps human-readable aliases to exact EnergyPlus variable, actuator, and meter identifiers.

### 2. The Deep Reinforcement Learning Agent (`drl/`)
The DRL folder implements a full reinforcement learning pipeline:
- **`eplus_sim.py`**: Contains the `EnergyPlusEnv` Gymnasium environment. It runs EnergyPlus in a daemon thread and synchronizes it with the main RL thread using threading events.
  - **Crucial step:** You must implement the `get_reward(step_data)` function here to translate the hackathon's scoring (Energy + CO2 + Temp penalties) into a scalar reward for the agent.
- **`train_drl.py`**: The training script. It configures the observation/action spaces (with min/max bounds for normalization), defines the SAC model hyperparameters (learning rate, batch size, gamma, etc.), and handles the optional Behavioral Cloning (BC) pre-training using data generated by the RBC controller.
- **`evaluate_drl.py`**: Loads a trained SAC model and runs it for a full test year, outputting a detailed CSV of observations and actions for analysis.
- **`variable_config.py`**: Similar to the RBC version, but structured as lists of tuples (`SENSOR_DEF`, `ACTUATOR_DEF`, `METER_DEF`) for programmatic space generation.

### 3. Analysis and Comparison
- **`visualize_output.ipynb`**: Found inside each approach's folder. Used to analyze the `eplusout.csv` output of a single run, generating plots for energy use, CO2 classification, temperature compliance, and calculating the final Total Cost.
- **`output_calcs_comparison.ipynb`**: Found in the project root. Takes outputs from multiple models (e.g., baseline vs. scheduled vs. your custom DRL) and compares them side-by-side to see which approach yields the lowest total cost.