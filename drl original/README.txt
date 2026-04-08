DRL (Deep Reinforcement Learning) — EnergyPlus Simulation
==========================================================

Train and evaluate a Deep Reinforcement Learning agent that controls
HVAC setpoints in an EnergyPlus building simulation.


TABLE OF CONTENTS
-----------------

  1. Prerequisites
  2. Setup
  3. Implement the Reward Function
  4. Configure the RL Algorithm
  5. (Optional) Behavioral Cloning Pre-training
  6. Train the Agent
  7. Evaluate the Trained Agent
  8. Analysing Results
  9. Recommended Tools
  10. Troubleshooting
  11. Code Overview


======================================================================
1. PREREQUISITES
======================================================================

1.1  Python
...........

Go to https://www.python.org/downloads/ and download Python
version 3.10.x - 3.12.10, 3.12.10 suggested. Choose the correct
installer for your operating system (Windows/macOS/Linux).
Newer versions (3.13+) may have compatibility issues with some
packages.

During installation, check "Add Python to PATH".
Verify:  python --version


1.2  EnergyPlus
...............

Download EnergyPlus v25.1 or v25.2 from https://energyplus.net/downloads
Use the default install location: C:\EnergyPlusV25-1-0

For v25.2 the folder name will be EnergyPlusV25-2-0 (or similar).


======================================================================
2. SETUP
======================================================================

2.1  Create a virtual environment
..................................

Open a terminal in the project root:

  python -m venv .venv

  # Activate:
  # Windows (PowerShell):
  .venv\Scripts\Activate.ps1
  # Windows (cmd):
  .venv\Scripts\activate.bat
  # macOS / Linux:
  source .venv/bin/activate

  pip install stable-baselines3 imitation gymnasium numpy pandas torch

  # For the analysis notebooks you also need:
  pip install matplotlib seaborn


2.2  Set the EnergyPlus path in eplus_sim.py
.............................................

Open eplus_sim.py and set ENERGYPLUS_DIR to your install folder:

  ENERGYPLUS_DIR = r"C:\EnergyPlusV25-1-0"
  (For v25.2: ENERGYPLUS_DIR = r"C:\EnergyPlusV25-2-0")

This folder must contain the pyenergyplus/ subfolder with the Python API.


2.3  Set file paths in drl.py
..............................

Open drl.py and update train_config:

  "eplus_idf_filename": r"C:\path\to\DOAS_wNeutralSupplyAir_wFanCoilUnits.idf",
  "weather_filename":   r"C:\path\to\FIN_TR_Tampere...epw",
  "eplus_output_path":  "drl/outputs/train",

The .idf file is in the project root folder (one level above drl/).
The .epw file is in the project root folder or under EnergyPlus WeatherData/.

Also set a different weather file for eval_config if you want to evaluate
on a different weather year (otherwise it can be the same).


======================================================================
3. IMPLEMENT THE REWARD FUNCTION
======================================================================

Open eplus_sim.py and find the get_reward() function (around line 330).
Currently it returns 0.0 (placeholder).

You must replace this with your own objective. The function receives
step_data, a dict with all sensor and meter readings for the current
timestep.


======================================================================
4. CONFIGURE THE RL ALGORITHM AND HYPERPARAMETERS
======================================================================

Open drl.py and find the SAC model definition.
Replace the xxx placeholders with actual values:

  sac_model = SAC(
      policy="MlpPolicy",
      env=train_env,
      learning_rate=3e-4,       # typical range: 1e-4 to 1e-3
      batch_size=256,           # typical: 64, 128, 256
      gamma=0.99,               # discount factor (0.95-0.999)
      tau=0.005,                # soft-update coefficient (0.001-0.01)
      ent_coef="auto",          # "auto" lets SAC tune entropy automatically
      verbose=1,
      tensorboard_log="drl/models/tb_logs_sac",
  )

  sac_model.learn(
      total_timesteps=500_000,  # depends on episode length and patience
      callback=eval_callback_sac,
      progress_bar=True,
  )

Typical starting-point values for HVAC control:

  Parameter         Suggested value     Notes
  ----------------------------------------------------------------
  learning_rate     3e-4                Lower if unstable
  batch_size        256                 256-512 works well for SAC
  gamma             0.99                High discount for long episodes
  tau               0.005               Soft target update rate
  ent_coef          "auto"              SAC auto-tunes exploration
  total_timesteps   500_000 - 2_000_000 More = better but slower

Observation/action spaces are already configured in train_config.
The agent controls 4 actuators (normalised to [-1, 1]):
  - cooling_setpoint   [18, 25] C
  - heating_setpoint   [18, 25] C
  - ahu_supply_temp    [16, 22] C
  - supply_fan_flow    [0.0, 0.96] kg/s


======================================================================
5. (OPTIONAL) BEHAVIORAL CLONING PRE-TRAINING
======================================================================

The training template uses Behavioral Cloning (BC) to pre-train the
agent before SAC fine-tuning. BC requires expert demonstration data.

1. Run one of the RBC controllers (e.g. rbc_scheduled) to generate a
   full-year simulation with trajectory logging:

     cd rbc_scheduled
     python run_idf.py

   This produces an expert_data.json file with observation-action pairs
   collected at every timestep.

2. Copy (or symlink) the expert data into the drl/ folder:

     copy ..\rbc_scheduled\eplus_out\expert_data.json expert_data.json

3. In train_drl.py, set EXPERT_JSON to point to this file:

     EXPERT_JSON = "expert_data.json"

If you want to skip BC pre-training, remove or comment out the BC
section in train_drl.py and train SAC from scratch.


======================================================================
6. TRAIN THE AGENT
======================================================================

Activate the virtual environment and run:

  cd drl
  python drl.py

Training progress is printed to the console and logged to TensorBoard:

  tensorboard --logdir drl/models/tb_logs_sac

The best model is saved automatically to drl/models/best_model_sac/
by the EvalCallback. The final model is saved to drl/models/sac_hvac.


======================================================================
7. EVALUATE THE TRAINED AGENT
======================================================================

Open eval_drl.py and update:

1. File paths in train_config (same idf, set eval weather file + output path):

     "eplus_idf_filename": r"C:\path\to\DOAS_wNeutralSupplyAir_wFanCoilUnits.idf",
     "weather_filename":   r"C:\path\to\eval_weather.epw",
     "eplus_output_path":  "drl/outputs/eval",

2. The model load path (line with SAC.load):

     model = SAC.load("drl/models/sac_hvac", env=env)

   Or point to the best model: "drl/models/best_model_sac/best_model"

Then run:

  python eval_drl.py

Results are saved to a timestamped CSV file in the drl/ folder, containing
observations, actions, and rewards for every simulation timestep.


======================================================================
8. ANALYSING RESULTS
======================================================================

8.1  visualize_output.ipynb  (this folder)
..........................................

Analyses the simulation output (eplusout.csv) for this approach.
Open it, select the .venv Python kernel, and Run All. Sections:

  - Energy:      Annual and monthly electricity/gas totals (kWh),
                 energy costs (EUR), and time-series plots.
  - CO2:         Zone CO2 concentrations, classification into S1/S2/S3
                 categories (90 % compliance rule), and penalty cost.
  - Temperature: Zone temperatures, classification into S1/S2/S3
                 categories (90 % compliance rule), and hourly penalty
                 cost.
  - Supply Air:  DOAS supply-air temperature and mass flow
                 rate over the year.
  - Total Cost:  Summary table with energy cost + CO2 penalty +
                 temperature penalty = total annual cost (EUR).


8.2  output_calcs_comparison.ipynb  (project root)
..................................................

Compares multiple approaches side by side. Edit the MODELS dict at
the top to point to the eplusout.csv files from each approach.
Sections:

  - Energy comparison:      Side-by-side annual totals, monthly bar
                            charts, and cost table for all models.
  - CO2 comparison:         CO2 classification and penalties per model.
  - Temperature comparison: Temperature classification and penalties
                            per model.
  - Total Cost:             Combined cost comparison table with
                            energy + CO2 + temperature penalties.


======================================================================
9. RECOMMENDED TOOLS
======================================================================

We recommend Visual Studio Code (VS Code) as the code editor.
Download from https://code.visualstudio.com/

VS Code is free, cross-platform, and has excellent Python + Jupyter
support.


9.1  Installing VS Code extensions
....................................

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
  File -> Close Window, then re-open VS Code.
  (Or press Ctrl+Shift+P -> type "Reload Window" -> press Enter.)

This ensures VS Code fully loads the new extensions.


9.2  Opening the project
.........................

  1. In VS Code: File -> Open Folder.
  2. Navigate to the project root folder (the folder that contains
     rbc_scheduled/, drl/, and the .idf file).
  3. Click "Select Folder" (Windows) or "Open" (macOS).
  4. If VS Code asks "Do you trust the authors of the files in this
     folder?" click "Yes, I trust the authors".


9.3  Selecting the Python interpreter (for scripts)
.....................................................

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


9.4  Selecting the notebook kernel (for .ipynb files)
......................................................

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
  - Restart VS Code (File -> Close Window, then re-open).
  - Try again from step 2.


9.5  Running scripts in the terminal
......................................

  1. Open the integrated terminal: Rightclick the wanted folder and choose 'Open in Integrated Terminal' or
     Terminal -> New Terminal from the menu bar.
  2. Make sure the virtual environment is activated. You should see
     (.venv) at the beginning of the terminal prompt. If not, run:

     # Windows (PowerShell):
     .venv\Scripts\Activate.ps1
     # Windows (cmd):
     .venv\Scripts\activate.bat
     # macOS / Linux:
     source .venv/bin/activate

  3. Navigate to the approach folder and run the script:

     cd drl
     python drl.py


9.6  Running notebooks
.......................

  1. Open a .ipynb file (double-click in the Explorer sidebar).
  2. Select the .venv kernel (see section 9.4 above).
  3. Click "Run All" at the top, or press Shift+Enter to run one
     cell at a time.


======================================================================
10. TROUBLESHOOTING
======================================================================

Problem: ModuleNotFoundError: No module named 'pyenergyplus'
  Fix:   ENERGYPLUS_DIR in eplus_sim.py is wrong or EnergyPlus is
         not installed.

Problem: ModuleNotFoundError: No module named 'stable_baselines3'
  Fix:   Activate the venv and run:
         pip install stable-baselines3 sb3-contrib gymnasium numpy pandas torch

Problem: EnergyPlus failed - aborting wait / Timed out
  Fix:   Check the output folder for eplusout.err. Common causes:
         wrong .idf path, wrong .epw path, or version mismatch.

Problem: Reward is always 0.0
  Fix:   You need to implement get_reward() in eplus_sim.py (Step 3).

Problem: Agent actions are all the same / not learning
  Fix:   Check that the reward function produces meaningful signal.
         Try increasing total_timesteps or adjusting learning_rate.

Problem: xxx placeholders cause errors in drl.py
  Fix:   Replace all xxx values with actual hyperparameters (Step 4).


======================================================================
11. CODE OVERVIEW
======================================================================

File overview
.............

  drl.py             - Training script. Configure hyperparameters here.
  eval_drl.py        - Evaluation script. Loads a trained model and
                       runs a test year.
  eplus_sim.py       - Gymnasium environment coupling EnergyPlus to
                       the RL agent.
  variable_config.py - Defines sensors, meters, and actuators.
  logger_eplus.py    - Logging setup (console + file).


eplus_sim.py
------------
Gymnasium environment that couples EnergyPlus to an RL agent.
EnergyPlus runs in a daemon thread; the agent runs in the main thread.
Synchronisation uses three threading.Event objects (obs_event, act_event,
stop_event) to hand off observations and actions each timestep.

Key components:

  - EnergyPlusEnv (gym.Env)
      Standard reset() / step(action) / close() interface.
      On reset(): launches EnergyPlus in a daemon thread, waits for the
      first observation. On step(): sends the RL action to E+, waits for
      the next observation, computes reward, checks termination.

  - callback_function_bp(state)
      Per-timestep E+ callback. Reads sensors -> signals obs_event ->
      waits for act_event -> applies action to actuators. Uses a safe
      default action during warm-up and a short post-warmup hold period.

  - run_energyplus(idf, weather, output_path, callback)
      Starts a full E+ simulation (blocking, runs inside daemon thread).
      Each episode writes to a unique run_N/ subdirectory.

  - get_observations(step_data, config)
      Extracts the observation vector from raw E+ data. Optionally
      min-max normalises to [-1, 1] using bounds from the config dict.

  - get_reward(step_data)
      Computes the scalar reward.

  - Helper functions: _init_handles(), _read_current_timestep(),
    _apply_action(), get_time().

train_drl.py
------------
SAC training script. Defines train_config and eval_config dicts with:
  - File paths (IDF, EPW, output directory)
  - Observation list (22 sensors: outdoor temp, 5x zone temp/RH/CO2,
    plenum temp, HVAC electricity, gas, hour, day of week)
  - Action list (4 actuators: heating/cooling setpoints, supply air
    temp, fan mass flow)
  - Observation/action min/max bounds for normalisation

Creates a Monitor-wrapped EnergyPlusEnv, builds an SAC model, trains
with EvalCallback for best-model checkpointing, and saves the final
model. Hyperparameters (learning_rate, batch_size, gamma, etc.) are
configured in the SAC constructor call.

evaluate_drl.py
---------------
Evaluation script. Loads a trained SAC model, runs one full-year
episode in the EnergyPlusEnv, and saves a timestep-level CSV with:
  - Normalised observations (all 22 sensors)
  - Normalised actions (4 actuators)
  - Next-step observations
  - Reward, done, truncated flags

Output CSV is timestamped (e.g. sac_bc_full_year_test_20260405_120000.csv).

variable_config.py
------------------
Defines EnergyPlus I/O point mappings in three lists:

  SENSOR_DEF  -  Read-only sensors. Built programmatically for 5 zones
                 (temp, RH, occupancy, CO2 per zone) plus outdoor temp,
                 plenum temp, and plenum CO2. Each entry is a tuple:
                 (alias, variable_name, key_value, request_flag).

  ACTUATOR_DEF - Writable control points (4 actuators):
                 heating setpoint, cooling setpoint, supply air temp
                 schedule, fan mass flow rate. Each entry:
                 (alias, component_type, control_type, key_value).

  METER_DEF   -  Cumulative energy meters (HVAC electricity, fan
                 electricity, cooling/heating coil transfer, net
                 facility electricity, natural gas). Each entry:
                 (alias, meter_name).

logger_eplus.py
---------------
Standard Python logging setup. Creates a named logger ("eplus_sim")
with two handlers:
  - Console (stdout): INFO level and above
  - File (eplus_sim.log): DEBUG level and above (captures everything)

Imported as: from logger_eplus import logger

drl_output/ (output files)
---------------------------
Same EnergyPlus outputs as the RBC approach (one set per episode run):

  eplusout.csv   -  Timestep-level output variables (main results).
  eplusmtr.csv   -  Timestep-level meter data (energy by end-use).
  eplustbl.htm   -  HTML summary report (annual totals, unmet hours).
  eplusout.sql   -  SQLite database with all output data.
  eplusout.err   -  Warnings and errors (check first if something fails).
  eplusout.eso   -  Raw time-series output (binary -> .csv).
  eplusout.mtr   -  Raw meter output (binary -> eplusmtr.csv).
  eplusout.eio   -  One-time info (design-day results, sizing).
  eplusout.rdd   -  Available output variables in the model.
  eplusout.mdd   -  Available meters in the model.
