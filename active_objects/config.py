import itertools
import os

# general TODO:
# 2. misc_functions.py true2ecc i ecc2mean se mogu vektorizovati

# simulation parameters
TIME_OF_SIMULATION = 1 # in years
EPOCH = 60980.0015818772

N_KINEMATICS = 1 # testing

# generator parameters
N0 = 0.1

D_REF = 100 # 100m
D_STEPS = [
    [10, 12], 
    [12, 15], 
    [15, 20], 
    [20, 25], 
    [25, 30], 
    [30, 40], 
    [40, 50], 
    [50, 70], 
    [70, 100], 
    [100, 100000]
]

# physical parameters
SDF_SLOPES = [-1.5, -2.0, -2.5, -3.0, -3.5, -4.0]
ACTIVITY_MODELS = ["Kresak", "Bailey & Stagg", "Weissman", "Sosa & Fernandez", "JPL"]

# pipeline settings
STARTING_CORE = 1
MAX_CORES = 1
BASE_ARCHIVE_DIR = "archive"
KINEMATICS_DIR = "kinematics_archive"

# creating the output archive folder
if not os.path.exists(BASE_ARCHIVE_DIR):
    os.makedirs(BASE_ARCHIVE_DIR)

# sorcha input filenames
INI_FILENAME = "active.ini"
ORBIT_FILENAME = "orbit.txt"
PP_FILENAME = "pp.txt"
COMPLEX_FILENAME = "cpp.txt"
PD_FILENAME = "baseline_v4.3.5_1_year.db"
SORCHA_OUTPUT_PREFIX = "sorcha_raw_output"

# preparing the configuration settings for all the runs
RUN_CONFIGS = []
task_id = 1

SDF_SLOPES = [-4.0]
ACTIVITY_MODELS = ["Kresak"]


# prepare configs for simulations
for kin_id in range(N_KINEMATICS):
    for alpha, model in itertools.product(SDF_SLOPES, ACTIVITY_MODELS):
        # e.g, 0001_kin01_sfd1.5_model1
        model_idx = ACTIVITY_MODELS.index(model)
        folder_name = f"{task_id:04d}_kin{kin_id:02d}_sfd{abs(alpha):.1f}_model{model_idx}"

        RUN_CONFIGS.append({
            "task_id": task_id,
            "kin_id": kin_id,
            "alpha": alpha,
            "activity_model": model,
            "model_idx": model_idx,
            "folder_name": folder_name
        })
        task_id += 1