import itertools
import os

# general TODO:
# 1. kde se ne mora računati dvaput za iste zvezde. adaptirati synth pop
# 2. misc_functions.py true2ecc i ecc2mean se mogu vektorizovati

# simulation parameters
TIME_OF_SIMULATION = 1 # in years
EPOCH = 60980.0015818772

N_KINEMATICS = 100
N_STARS_PER_SAMPLE = 1000
# TODO dodati procente, zapisati ID
# TODO podeliti na tri generisanja

# generator parameters
N0 = 0.01

D_REF = 100 # 100m
D_STEPS = [[10,100], [100, 100000]] # 10m - 100km

# physical parameters
SDF_SLOPES = [-1.5, -2.0, -2.5, -3.0, -3.5, -4.0]
ALBEDOS = [0.05, 0.1, 0.2, "trojan"]

# pipeline settings
STARTING_CORE = 1
MAX_CORES = 4
BASE_ARCHIVE_DIR = "archive"
KINEMATICS_DIR = "kinematics_archive"

# TEST
N_KINEMATICS = 1
MAX_CORES = 1
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
SDF_SLOPES = [-4.0]
ALBEDOS = [0.2]


# creating the output archive folder
if not os.path.exists(BASE_ARCHIVE_DIR):
    os.makedirs(BASE_ARCHIVE_DIR)

# sorcha input filenames
INI_FILENAME = "inactive.ini"
ORBIT_FILENAME = "orbit.txt"
PP_FILENAME = "pp.txt"
PD_FILENAME = "baseline_v4.3.5_1_year.db"
SORCHA_OUTPUT_PREFIX = "sorcha_raw_output"

# preparing the configuration settings for all the runs
RUN_CONFIGS = []
task_id = 1

# prepare configs for simulations
for kin_id in range(N_KINEMATICS):
    for alpha, albedo in itertools.product(SDF_SLOPES, ALBEDOS):
        # e.g, 0001_kin01_sfd1.5_alb0.10
        folder_name = f"{task_id:04d}_kin{kin_id:02d}_sfd{abs(alpha):.1f}_alb{albedo}"

        RUN_CONFIGS.append({
            "task_id": task_id,
            "kin_id": kin_id,
            "alpha": alpha,
            "albedo": albedo,
            "folder_name": folder_name
        })
        task_id += 1