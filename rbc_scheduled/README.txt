RBC Scheduled — EnergyPlus Simulation
======================================

Rule-based controller with outdoor-compensated setpoints, schedule-based
fan operation, and CO2 demand-controlled ventilation.


TABLE OF CONTENTS
-----------------

  1. Prerequisites
  2. Setup
  3. Running the Simulation
  4. Analysing Results
  5. Recommended Tools
  6. Troubleshooting
  7. Code Overview


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

During installation:
  - Windows: check "Add Python to PATH".
  - macOS/Linux: Python is often pre-installed. If not, use your
    package manager (e.g. brew install python, sudo apt install python3).

Verify in a terminal:

  python --version          # or python3 --version on some systems


1.2  EnergyPlus
...............

Download EnergyPlus v25.1 or v25.2 from https://energyplus.net/downloads
(installers are available for Windows, macOS, and Linux).

Run the installer. Default install locations (example for v25.1):

  Windows : C:\EnergyPlusV25-1-0
  macOS   : /Applications/EnergyPlus-25-1-0
  Linux   : /usr/local/EnergyPlus-25-1-0

For v25.2 the folder name will be EnergyPlusV25-2-0 (or similar).

Note the installation path - you will need it in step 2.2.
The folder must contain the energyplus executable and a subfolder
called pyenergyplus/.


======================================================================
2. SETUP
======================================================================

2.1  Create a virtual environment
..................................

Open a terminal in the project root folder (the folder containing
rbc_scheduled/, the .idf file, etc.) and run:

  python -m venv .venv

  # Activate:
  # Windows (PowerShell):
  .venv\Scripts\Activate.ps1
  # Windows (cmd):
  .venv\Scripts\activate.bat
  # macOS / Linux:
  source .venv/bin/activate

  pip install numpy

  # For the analysis notebooks you also need:
  pip install pandas matplotlib seaborn ipykernel


2.2  Set the EnergyPlus path in run_idf.py
...........................................

Open run_idf.py and update the ENERGYPLUS_DIR variable to match
your installation path (adjust the version number as needed):

  # Windows (v25.1)
  ENERGYPLUS_DIR = r"C:\EnergyPlusV25-1-0"
  # Windows (v25.2)
  ENERGYPLUS_DIR = r"C:\EnergyPlusV25-2-0"

  # macOS (v25.1)
  ENERGYPLUS_DIR = "/Applications/EnergyPlus-25-1-0"
  # macOS (v25.2)
  ENERGYPLUS_DIR = "/Applications/EnergyPlus-25-2-0"

  # Linux (v25.1)
  ENERGYPLUS_DIR = "/usr/local/EnergyPlus-25-1-0"
  # Linux (v25.2)
  ENERGYPLUS_DIR = "/usr/local/EnergyPlus-25-2-0"

This path is added to sys.path so Python can find the pyenergyplus API.


2.3  Set the simulation file paths in run_idf.py
..................................................

In the same file, update these three paths. Replace <PROJECT_ROOT>
with the absolute path to your project folder:

  IDF_FILE = Path("<PROJECT_ROOT>/DOAS_wNeutralSupplyAir_wFanCoilUnits.idf")
  EPW_FILE = Path("<PROJECT_ROOT>/FIN_TR_Tampere.Satakunnankatu.027440_TMYx.2004-2018.epw")
  OUT_DIR  = Path("<PROJECT_ROOT>/rbc_scheduled/eplus_out")

  # Example (macOS / Linux):
  #   IDF_FILE = Path("../DOAS_wNeutralSupplyAir_wFanCoilUnits.idf")
  # Example (Windows):
  #   IDF_FILE = Path(r"..\DOAS_wNeutralSupplyAir_wFanCoilUnits.idf")

IDF_FILE  - The .idf building model (included in the project root).
EPW_FILE  - The .epw weather file (included in the project root, or
            use the one shipped with EnergyPlus under WeatherData/).
OUT_DIR   - Output folder for simulation results (created automatically).


======================================================================
3. RUNNING THE SIMULATION
======================================================================

Make sure the virtual environment is activated, then:

  cd rbc_scheduled        # from the project root
  python run_idf.py

If everything is configured correctly you will see:

  Starting EnergyPlus with args: ['-d', '.../eplus_out', '-w', '...epw', '-r', '...idf']
  ...
  Simulation completed successfully. Output in: .../eplus_out

The results (CSV, SQL, HTML report, etc.) will be in the OUT_DIR folder.


======================================================================
4. ANALYSING RESULTS
======================================================================

4.1  visualize_output.ipynb  (this folder)
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


4.2  output_calcs_comparison.ipynb  (project root)
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
5. RECOMMENDED TOOLS
======================================================================

We recommend Visual Studio Code (VS Code) as the code editor.
Download from https://code.visualstudio.com/

VS Code is free, cross-platform, and has excellent Python + Jupyter
support.


5.1  Installing VS Code extensions
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


5.2  Opening the project
.........................

  1. In VS Code: File -> Open Folder.
  2. Navigate to the project root folder (the folder that contains
     rbc_scheduled/, drl/, and the .idf file).
  3. Click "Select Folder" (Windows) or "Open" (macOS).
  4. If VS Code asks "Do you trust the authors of the files in this
     folder?" click "Yes, I trust the authors".


5.3  Selecting the Python interpreter (for scripts)
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


5.4  Selecting the notebook kernel (for .ipynb files)
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


5.5  Running scripts in the terminal
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

     cd rbc_scheduled
     python run_idf.py


5.6  Running notebooks
.......................

  1. Open a .ipynb file (double-click in the Explorer sidebar).
  2. Select the .venv kernel (see section 5.4 above).
  3. Click "Run All" at the top, or press Shift+Enter to run one
     cell at a time.


======================================================================
6. TROUBLESHOOTING
======================================================================

Problem: ModuleNotFoundError: No module named 'pyenergyplus'
  Fix:   ENERGYPLUS_DIR is wrong or EnergyPlus is not installed.

Problem: ERROR: IDF file not found
  Fix:   Check that IDF_FILE points to an existing .idf file.

Problem: ERROR: Weather file not found
  Fix:   Check that EPW_FILE points to an existing .epw file.

Problem: Simulation fails with return code not equal to 0
  Fix:   Check eplus_out/eplusout.err for EnergyPlus error messages.

Problem: 'python' is not recognized / command not found
  Fix:   Try python3 instead, or check that Python is on your PATH.


======================================================================
7. CODE OVERVIEW
======================================================================

rbc_model.py
------------
The control strategy. Implements the RBCModel class with four methods:

  - zone_setpoints(outdoor_temp)
      Computes heating/cooling temperature setpoints using outdoor-
      compensated curves that track inside the S1 comfort band.
      Enforces a deadband to prevent simultaneous heating and cooling.

  - return_air_compensation(return_air_temp)
      Linearly maps return-air temperature to a supply-air temperature
      setpoint (19 C when return air is cool, 17 C when warm).

  - co2_flow_control(hour, day, co2, outdoor_temp)
      Determines the AHU fan mass flow rate. Uses CO2 demand-controlled
      ventilation: linearly ramps flow from FLOW_LOW to FLOW_BOOST as
      CO2 rises from 500 to 750 ppm.

  - compute_s1_limits(outdoor_temp)
      Helper that returns the S1 class lower/upper temperature limits
      as a function of outdoor temperature.

energyplus_controller.py
------------------------
Bridges the EnergyPlus Python API and the RBC model.

  - initialize_handles(state)
      Requests runtime handles for all variables, actuators, and meters
      defined in variable_config.py. Called once after warm-up.

  - control_callback(state)
      Called every zone timestep. Reads sensor data (zone temps, CO2,
      outdoor conditions, time-of-day), calls the RBC model to compute
      setpoints and flow rate, then writes them back to EnergyPlus.
      Also collects observation/action pairs for trajectory export.

  - get_variable(name, state) / set_actuator(name, value, state)
      Convenience wrappers for reading sensors and writing actuators.

variable_config.py
------------------
Defines the EnergyPlus I/O point mappings used by the controller.
Three dictionaries map short aliases to EnergyPlus identifiers:

  VARIABLES  -  Read-only sensors (outdoor temp, zone temps, RH, CO2,
                occupancy for each of the 5 zones + plenum).
  ACTUATORS  -  Writable control points (heating setpoint, cooling
                setpoint, supply air temp schedule, fan mass flow).
  METERS     -  (Cumulative) energy meters (HVAC electricity, gas,
                cooling/heating coil transfer, fan electricity).

run_idf.py
----------
Entry point. Sets up paths, creates the EnergyPlus API instance,
instantiates RBCModel + EnergyPlusController, registers three callbacks
(handle init, control logic, API data dump), and launches the simulation.
After the run it saves collected trajectories to expert_data.json.

eplus_out/ (output files)
-------------------------
EnergyPlus writes these files after a successful simulation:

  eplusout.csv   -  Timestep-level output variables (the main results file).
  eplusmtr.csv   -  Timestep-level meter data (energy by end-use).
  eplustbl.htm   -  HTML summary report (annual totals, unmet hours, etc.).
  eplusout.sql   -  SQLite database with all output data.
  eplusout.err   -  Simulation warnings and errors (check this first).
  eplusout.eso   -  Raw time-series output (binary, used to produce .csv).
  eplusout.mtr   -  Raw meter output (binary, used to produce eplusmtr.csv).
  eplusout.eio   -  One-time information (design-day results, sizing).
  eplusout.rdd   -  List of all available output variables in the model.
  eplusout.mdd   -  List of all available meters in the model.
  eplusout.bnd   -  Node/branch connection report.
  eplusout.shd   -  Surface shadowing combinations.
  eplusout.audit -  Input-file processing log.
  eplusout.end   -  One-line success/failure summary.
