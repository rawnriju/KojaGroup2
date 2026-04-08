HVAC CONTROL HACKATHON
======================

This repository contains template code for controlling a DOAS
(Dedicated Outdoor Air System) building model in EnergyPlus using
three approaches:

  rbc_full_on/    –  RBC: Full-On (ventilation always at max capacity)
  rbc_scheduled/  –  RBC: Scheduled Operation (time + CO2 demand control)
  drl/            –  Deep Reinforcement Learning (DRL) agent

All three control the same 5-zone small-office building model (.idf)
and share common EnergyPlus sensor/actuator definitions.


OBJECTIVE
---------

Design and implement an HVAC controller that balances three competing
objectives:

  1. ENERGY EFFICIENCY — minimise electricity and gas consumption
  2. THERMAL COMFORT  — keep zone temperatures inside comfort bands
  3. AIR QUALITY      — keep CO2 concentrations below thresholds

The winner is the team with the lowest TOTAL COST (energy + penalties).


SCORING
-------

Total cost = Energy cost + CO2 penalty + Temperature penalty

1. Energy cost

   Electricity : 0.11 €/kWh
   Natural gas : 0.06 €/kWh

   Both are computed from the EnergyPlus output meters
   (Elec_Facility_E_J and Gas_Facility_E_J).

2. CO2 penalty (per space, per hour above threshold)

   > 770 ppm but ≤ 970 ppm  :   2 €/h
   > 970 ppm but ≤ 1220 ppm :  10 €/h
   > 1220 ppm               :  50 €/h

   Applied independently to each of the 5 zones.
   The CO2 classification uses the 90% rule:
     S1 : ≥ 90% of hours below 770 ppm
     S2 : ≥ 90% of hours below 970 ppm
     S3 : ≥ 90% of hours below 1220 ppm

3. Temperature penalty (per space, per hour outside class band)

   Outside S1 but inside S2 :  1 €/h
   Outside S2 but inside S3 :  5 €/h
   Outside S3               : 25 €/h

   Temperature class bands are based on a 24-hour rolling outdoor
   temperature average (Finnish indoor climate classification S1/S2/S3).

   The classification uses the 90% rule:
     S1 : ≥ 90% of hours inside S1 band
     S2 : ≥ 90% of hours inside S2 band
     S3 : ≥ 90% of hours inside S3 band

   S1 limits (outdoor-temp-dependent):
     Lower: 20.5 °C (outdoor ≤ 0 °C) → 22.0 °C (outdoor ≥ 20 °C)
     Upper: 22.0 °C (outdoor ≤ 0 °C) → 25.0 °C (outdoor ≥ 15 °C)

   S2 limits:
     Lower: 20.5 °C (outdoor ≤ 0 °C) → 21.0 °C (outdoor ≥ 20 °C)
     Upper: 23.0 °C (outdoor ≤ 0 °C) → 26.0 °C (outdoor ≥ 15 °C)

   S3 limits:
     Lower: 20.0 °C (constant)
     Upper: 25.0 °C (outdoor ≤ 10 °C) → 27.0 °C (outdoor > 10 °C)


THE BUILDING MODEL
------------------

The building model is a DOE reference small-office with 5 conditioned
thermal zones (SPACE1-1 … SPACE5-1) and one return-air plenum (PLENUM-1).

HVAC system:
  - Dedicated Outdoor Air System (DOAS) with a central supply fan
  - Fan coil units (FCU) in each zone for local heating/cooling
  - Heat exchanger, chiller, cooling tower, boiler, pumps
  - CO2 generation from occupants in each zone

Building model file:
  DOAS_wNeutralSupplyAir_wFanCoilUnits.idf

Weather files (included in the project):
  FIN_TR_Tampere.Satakunnankatu.027440_TMYx.2004-2018.epw

Simulation timestep: 15 minutes (96 steps per day).
Simulation period: full calendar year.


CONTROLLABLE ACTUATORS
----------------------

Your controller writes 4 values at each timestep:

  Actuator              Physical range     Description
  -------------------------------------------------------------------
  heating_setpoint      18.0 – 25.0 °C     Zone heating temperature setpoint
  cooling_setpoint      18.0 – 25.0 °C     Zone cooling temperature setpoint
  ahu_supply_temp       16.0 – 21.0 °C     AHU supply air temperature
  supply_fan_flow        0.0 –  1.0 kg/s   AHU supply fan mass flow rate

Note: heating_setpoint must be ≤ cooling_setpoint (enforced by the
simulation environment for DRL; your RBC code should ensure this).


AVAILABLE OBSERVATIONS
----------------------

The following sensor readings are available at each timestep:

  Environment:
    outdoor_temp       — outdoor dry-bulb temperature [°C]
    plenum_temp        — return-air plenum temperature [°C]

  Per zone (5 zones: space1 … space5):
    space{i}_temp      — zone air temperature [°C]
    space{i}_rh        — zone relative humidity [%]
    space{i}_co2       — zone CO2 concentration [ppm]
    space{i}_occ       — zone occupant count (people)

  Energy meters:
    electricity_hvac   — HVAC electricity [J per timestep]
    fans_electricity   — fan electricity [J per timestep]
    cooling_energy     — cooling coil energy transfer [J]
    heating_energy     — heating coil energy transfer [J]
    gas_total          — natural gas consumption [J per timestep]
    facility_electricity_net — net facility electricity [J]

  Time:
    hour               — hour of day (0–23)
    day_of_week        — day of week (1 = Sunday … 7 = Saturday)

All sensors are defined in variable_config.py (each approach folder).
You can add or remove observations as needed.


GETTING THE TEMPLATE
====================

Option A — Clone with Git
--------------------------

   git clone https://github.com/koja-code/hackathon2026-materials
   cd hackathon2026-materials


Option B — Download as ZIP
---------------------------

1. Go to https://github.com/koja-code/hackathon2026-materials
2. Click the green "Code" button → "Download ZIP".
3. Extract the ZIP to a folder of your choice.
4. Open a terminal in that folder.


PROJECT STRUCTURE
=================

  <project-root>/
  ├── README.txt                       ← this file
  ├── DOAS_wNeutralSupplyAir_wFanCoilUnits.idf
  ├── FIN_TR_Tampere.*.epw
  │
  ├── rbc_full_on/                     ← Baseline: always max ventilation
  │   ├── rbc_model_1.py              ← control strategy (edit this)
  │   ├── energyplus_controller.py    ← E+ API bridge
  │   ├── run_idf.py                  ← entry point
  │   ├── variable_config.py          ← sensor/actuator definitions
  │   └── visualize_output.ipynb      ← analysis notebook
  │
  ├── rbc_scheduled/                   ← Template: scheduled + CO2 control
  │   ├── rbc_model.py                ← control strategy (edit this)
  │   ├── energyplus_controller.py    ← E+ API bridge
  │   ├── run_idf.py                  ← entry point
  │   ├── variable_config.py          ← sensor/actuator definitions
  │   └── visualize_output.ipynb      ← analysis notebook
  │
  ├── drl/                             ← Deep Reinforcement Learning
  │   ├── eplus_sim.py                ← Gymnasium environment
  │   ├── train_drl.py                ← training script (edit this)
  │   ├── evaluate_drl.py             ← evaluation script
  │   ├── variable_config.py          ← sensor/actuator definitions
  │   ├── logger_eplus.py             ← logging setup
  │   └── visualize_output.ipynb      ← analysis notebook
  │
  └── output_calcs_comparison.ipynb    ← side-by-side comparison


APPROACHES
==========

You can choose one or combine approaches:


A) RULE-BASED CONTROL (RBC)
----------------------------

Two starting templates are provided:

  rbc_full_on/rbc_model_1.py
    Simple baseline — fixed setpoints (22.5 °C heating & cooling),
    fixed supply air (19 °C), fan always at maximum (1.0 kg/s).
    No scheduling, no compensation, no CO2 control.

  rbc_scheduled/rbc_model.py
    Smarter template — outdoor-compensated zone setpoints, return-air-
    compensated supply temperature, CO2 demand-controlled ventilation
    (linear ramp 500–750 ppm), time-of-day fan scheduling.

To implement your own RBC:
  1. Open the rbc_model.py (or rbc_model_1.py) in your chosen folder.
  2. Modify the calculate_setpoints() method.
  3. All sensor data is available as function arguments.
  4. Return (heating_setpoint, cooling_setpoint, supply_air_temp, fan_flow).

To run:
  cd rbc_full_on      # or rbc_scheduled
  python run_idf.py

Results appear in eplus_out/eplusout.csv.


B) DEEP REINFORCEMENT LEARNING (DRL)
--------------------------------------

The DRL template uses Stable-Baselines3 with a SAC (Soft Actor-Critic)
algorithm. The EnergyPlus simulation runs as a Gymnasium environment.

To implement DRL:

  1. DESIGN THE REWARD FUNCTION
     Open drl/eplus_sim.py, find get_reward() (returns 0.0 by default).
     Replace with your own objective. The function receives step_data
     containing all sensor/meter readings.

     Good starting point: weighted negative sum of energy consumption,
     thermal comfort deviation, and CO2 violation.

  2. CONFIGURE OBSERVATION SPACE (optional)
     Open drl/train_drl.py and edit OBS_SPEC to add/remove sensors.
     The min/max bounds and config arrays are built automatically.

  3. SET HYPERPARAMETERS
     In drl/train_drl.py, configure the SAC model constructor:
       learning_rate
       batch_size
       gamma
       total_timesteps

  4. TRAIN
     cd drl
     python train_drl.py

     Monitor with TensorBoard:
     tensorboard --logdir models/tb_logs_sac

  5. EVALUATE
     python evaluate_drl.py

     Results saved to a timestamped CSV.

The trained model is saved to drl/models/sac_bc_hvac.


PREREQUISITES & SETUP
=====================

Software:
  - Python 3.10.x – 3.12.10  (https://www.python.org/downloads/)
    Newer versions (3.13+) may have compatibility issues.
  - EnergyPlus v25.1 or v25.2  (https://energyplus.net/downloads)

1. Go to https://www.python.org/downloads/ and download Python
   version 3.10.x – 3.12.10, 3.12.10 suggested. Choose the correct installer for your
   operating system (Windows/macOS/Linux).
   - Windows: check "Add Python to PATH" during installation.
   - macOS/Linux: use your package manager if not pre-installed.

2. Install EnergyPlus v25.1 or v25.2.
   Default install paths (example for v25.1):
     Windows : C:\EnergyPlusV25-1-0
     macOS   : /Applications/EnergyPlus-25-1-0
     Linux   : /usr/local/EnergyPlus-25-1-0

3. Create and activate a virtual environment:

   python -m venv .venv

   # Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # Windows (cmd):
   .venv\Scripts\activate.bat
   # macOS / Linux:
   source .venv/bin/activate

4. Install packages:

   # For RBC only:
   pip install numpy

   # For DRL:
   pip install stable-baselines3 imitation gymnasium numpy pandas torch

   # For analysis notebooks:
   pip install matplotlib seaborn

5. Set the EnergyPlus path:
   - RBC: open rbc_*/run_idf.py → set ENERGYPLUS_DIR
   - DRL: open drl/eplus_sim.py → set ENERGYPLUS_DIR

6. Set the simulation file paths:
   - RBC: open rbc_*/run_idf.py → set IDF_FILE, EPW_FILE, OUT_DIR
   - DRL: open drl/train_drl.py → set IDF_FILE, WEATHER_FILE

See rbc_full_on/README.txt, rbc_scheduled/README.txt, or drl/README.txt
for detailed step-by-step instructions.


RECOMMENDED TOOLS
=================

Code editor: Visual Studio Code (VS Code)
------------------------------------------

Download from https://code.visualstudio.com/

VS Code is free, cross-platform, and has excellent Python + Jupyter
support.


Installing VS Code extensions
...............................

You need two extensions: Python and Jupyter.

  1. Open VS Code.
  2. Click the Extensions icon in the left sidebar (it looks like
     four small squares, or press Ctrl+Shift+X).
  3. In the search bar at the top, type:  Python
  4. Find "Python" by Microsoft (ms-python.python) and click Install.
  5. Wait for the installation to finish.
  6. Search again for:  Jupyter
  7. Find "Jupyter" by Microsoft (ms-toolsai.jupyter) and click Install.
  8. Wait for the installation to finish.

After installing both extensions, restart VS Code:
  File → Close Window, then re-open VS Code.
  (Or press Ctrl+Shift+P → type "Reload Window" → press Enter.)

This ensures VS Code fully loads the new extensions.


Opening the project
....................

  1. In VS Code: File → Open Folder.
  2. Navigate to the project root folder (the folder that contains
     rbc_scheduled/, drl/, and the .idf file).
  3. Click "Select Folder" (Windows) or "Open" (macOS).
  4. If VS Code asks "Do you trust the authors of the files in this
     folder?" click "Yes, I trust the authors".


Selecting the Python interpreter (for scripts)
................................................

VS Code needs to know which Python to use. You want it to use the
virtual environment (.venv) where you installed all the packages.

  1. Press Ctrl+Shift+P to open the Command Palette.
  2. Type:  Python: Select Interpreter
  3. Press Enter.
  4. You will see a list of Python installations. Choose the one
     that shows ".venv" in its path, for example:

       Python 3.12.10 ('.venv': venv)  .venv\Scripts\python.exe

  5. If you do not see .venv in the list:
     a. Click "Enter interpreter path..." at the top.
     b. Navigate to: <project-root>\.venv\Scripts\python.exe
        (Windows) or <project-root>/.venv/bin/python (macOS/Linux).
     c. Select it.

  The selected interpreter appears in the bottom-right corner of
  VS Code (or in the bottom status bar).


Selecting the notebook kernel (for .ipynb files)
..................................................

When you open a Jupyter notebook (.ipynb file), you need to tell
VS Code which Python kernel to use for running the cells.
Check the pictures from the end of the Task Description pdf. 

  1. Double-click a .ipynb file in the Explorer sidebar to open it.
  2. In the top-right corner of the notebook, click "Select Kernel"
     (or it may show "No Kernel" or a different Python version).
  3. Choose "Python Environments..." from the dropdown.
  4. Select the .venv interpreter, for example:

       Python 3.12.10 ('.venv': venv)  .venv\Scripts\python.exe

  5. The kernel name in the top-right should now show ".venv (Python 3.12.10)"
     or similar.
  6. Now you can run cells with Shift+Enter or click "Run All" at the top.

  If the .venv kernel does not appear:
  - Make sure you installed ipykernel in the venv:
      pip install ipykernel
  - Restart VS Code (File → Close Window, then re-open).
  - Try again from step 2.


Running simulations (Python scripts)
--------------------------------------

Use the VS Code integrated terminal (Rightclick the wanted folder and choose 'Open in Integrated Terminal' or Terminal → New Terminal):

  1. Make sure the virtual environment is activated (you should see
     (.venv) in the terminal prompt). If not, run:

     # Windows (PowerShell):
     .venv\Scripts\Activate.ps1
     # Windows (cmd):
     .venv\Scripts\activate.bat
     # macOS / Linux:
     source .venv/bin/activate

  2. Navigate to the approach folder and run:

     # RBC
     cd rbc_full_on          # or rbc_scheduled
     python run_idf.py

     # DRL training
     cd drl
     python train_drl.py

     # DRL evaluation
     cd drl
     python evaluate_drl.py

  The simulation takes a few minutes. Output appears in the terminal.
  EnergyPlus results are saved to the output folder (eplus_out/ or
  outputs/).


Running analysis notebooks (.ipynb)
-------------------------------------

Each approach folder contains a visualize_output.ipynb notebook for
single-model analysis. The project root contains
output_calcs_comparison.ipynb for comparing multiple models.

To run notebooks in VS Code:

  1. Open the .ipynb file in VS Code (double-click it in the explorer).
  2. VS Code will show the notebook with code cells and markdown cells.
  3. Select the Python kernel: click "Select Kernel" in the top-right
     and choose the .venv interpreter (see section above).
  4. Run cells with Shift+Enter, or click "Run All" at the top.

Workflow:
  1. Run a simulation first (python run_idf.py or train/evaluate DRL).
  2. Open the per-model notebook:
       rbc_full_on/visualize_output.ipynb
       rbc_scheduled/visualize_output.ipynb
       drl/visualize_output.ipynb
  3. Run all cells to see energy, CO2, temperature, and cost analysis.
  4. Open output_calcs_comparison.ipynb to compare models side by side.
     Edit the MODELS dict at the top to point to the correct
     eplusout.csv paths for each approach you want to compare.


ANALYSING RESULTS
=================

Each approach folder contains a visualize_output.ipynb notebook that
analyses that approach's simulation output (eplusout.csv). The notebook
has the following sections:

  - Energy:      Annual and monthly electricity/gas totals (kWh),
                 energy costs (€), and time-series plots.
  - CO2:         Zone CO2 concentrations, classification into S1/S2/S3
                 categories (90 % compliance rule), and penalty cost.
  - Temperature: Zone temperatures, classification into S1/S2/S3
                 categories (90 % compliance rule), and hourly penalty
                 cost.
  - Supply Air:  DOAS supply-air temperature and mass flow
                 rate over the year.
  - Total Cost:  Summary table with energy cost + CO2 penalty +
                 temperature penalty = total annual cost (€).

To compare multiple approaches side by side, use the root-level
output_calcs_comparison.ipynb notebook. Edit the MODELS dict at the
top to point to the eplusout.csv files from each approach. It contains:

  - Energy comparison:      Side-by-side annual totals, monthly bar
                            charts, and cost table for all models.
  - CO2 comparison:         CO2 classification and penalties per model.
  - Temperature comparison: Temperature classification and penalties
                            per model.
  - Total Cost:             Combined cost comparison table with
                            energy + CO2 + temperature penalties.


TROUBLESHOOTING
===============

Problem: ModuleNotFoundError: No module named 'pyenergyplus'
  Fix:   ENERGYPLUS_DIR is wrong or EnergyPlus is not installed.

Problem: ERROR: IDF file not found
  Fix:   Check that IDF_FILE / eplus_idf_filename points to the .idf.

Problem: ERROR: Weather file not found
  Fix:   Check that EPW_FILE / weather_filename points to the .epw.

Problem: Simulation fails (non-zero return code)
  Fix:   Check eplus_out/eplusout.err for EnergyPlus error messages.

Problem: 'python' is not recognized / command not found
  Fix:   Try python3, or check that Python is on your PATH.

Problem: ModuleNotFoundError: No module named 'stable_baselines3'
  Fix:   pip install stable-baselines3 sb3-contrib gymnasium numpy pandas torch

Problem: DRL reward is always 0.0
  Fix:   You need to implement get_reward() in drl/eplus_sim.py.

Problem: DRL agent actions are all the same / not learning
  Fix:   Check reward function and try adjusting hyperparameters.


CODE OVERVIEW
=============

RBC (rbc_full_on/ and rbc_scheduled/)
--------------------------------------

  rbc_model.py / rbc_model_1.py
    The control strategy. Implements calculate_setpoints() which receives
    sensor readings (zone temp, outdoor temp, return air temp, occupancy,
    hour, day, CO2) and returns 4 control values (heating setpoint,
    cooling setpoint, supply air temp, fan flow rate).

    rbc_full_on: all values are fixed (baseline).
    rbc_scheduled: outdoor-compensated setpoints, return-air-compensated
    supply temp, CO2 demand-controlled ventilation, time-of-day scheduling.

  energyplus_controller.py
    Bridges the EnergyPlus Python API and the RBC model. Reads sensor
    data via runtime handles, calls the model, writes actuator values.
    Also collects observation/action pairs for trajectory export.

  variable_config.py
    Maps short aliases to EnergyPlus variable/actuator/meter identifiers.
    Three dicts: VARIABLES (sensors), ACTUATORS (control points),
    METERS (energy meters).

  run_idf.py
    Entry point. Sets up paths, instantiates the model and controller,
    registers EnergyPlus callbacks, launches the simulation.


DRL (drl/)
-----------

  eplus_sim.py
    Gymnasium environment coupling EnergyPlus to the RL agent.
    E+ runs in a daemon thread; the agent runs in the main thread.
    Synchronisation via threading.Event objects (obs/act/stop).

    Key parts:
      EnergyPlusEnv — gym.Env with reset()/step()/close()
      callback_function_bp — per-timestep E+ callback
      get_observations — builds observation vector from sensor data
      get_reward — YOUR reward function (returns 0.0 by default)

  train_drl.py
    Training script. Defines OBS_SPEC and ACTION_SPEC dicts that
    configure the observation and action spaces. Includes behavioral
    cloning pre-training from expert trajectories, followed by SAC
    fine-tuning with weight transfer.

  evaluate_drl.py
    Loads a trained model, runs a full-year evaluation, saves results
    to a timestamped CSV. Imports OBS_SPEC/ACTION_SPEC from train_drl.

  variable_config.py
    Sensor, meter, and actuator definitions (same role as RBC version
    but uses list-of-tuples format: SENSOR_DEF, METER_DEF, ACTUATOR_DEF).

  logger_eplus.py
    Logging configuration (console INFO + file DEBUG).


ENERGYPLUS OUTPUT FILES
-----------------------

After a successful simulation, the output folder contains:

  eplusout.csv   — timestep-level output variables (main results)
  eplusmtr.csv   — timestep-level meter data (energy by end-use)
  eplustbl.htm   — HTML summary report (annual totals, unmet hours)
  eplusout.sql   — SQLite database with all output data
  eplusout.err   — warnings and errors (check this first if something fails)
  eplusout.rdd   — list of all available output variables in the model
  eplusout.mdd   — list of all available meters in the model


NOTES
=====

- The project root folder can be named anything. All internal
  references use relative paths.

- The .idf and .epw files in the project root are shared by all
  approaches. You can also use weather files from the EnergyPlus
  installation (typically under WeatherData/).

- Each approach writes EnergyPlus output to its own subfolder
  (rbc_full_on/eplus_out/, rbc_scheduled/eplus_out/, drl/outputs/).

- After running, use output_calcs_comparison.ipynb to compare
  energy, CO2, and temperature performance side by side.
