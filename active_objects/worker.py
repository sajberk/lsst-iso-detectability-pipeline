import os
import subprocess
import traceback
import datetime
import time
import shutil
import csv
import numpy as np

from config import (
    BASE_ARCHIVE_DIR,
    INI_FILENAME,
    PP_FILENAME,
    COMPLEX_FILENAME,
    ORBIT_FILENAME,
    PD_FILENAME,
    SORCHA_OUTPUT_PREFIX
)

import gen_cometary

def run_single_batch(config):
    """Processes a single configuration and returns, freeing the core for the next task."""

    # a nice timer
    batch_start_time = time.time()

    # proc the rng using the unique task_id so every batch is perfectly distinct and reproducible
    seed_seq = np.random.SeedSequence(config["task_id"])
    np.random.seed(seed_seq.generate_state(1)[0])

    # preparing the output folder for a single core
    folder_name = config["folder_name"]
    config_dir = os.path.join(BASE_ARCHIVE_DIR, folder_name)
    os.makedirs(config_dir, exist_ok=True)
    
    # find the next available batch number
    batch_num = 1
    batch_dir_name = f"batch_{batch_num:04d}"
    batch_path = os.path.join(config_dir, batch_dir_name)

    # increment if this batch already exists (safe restarts)
    while os.path.exists(batch_path):
        batch_num += 1
        batch_dir_name = f"batch_{batch_num:04d}"
        batch_path = os.path.join(config_dir, batch_dir_name)


    # making the temporary directory for this sorcha run
    temp_batch_path = os.path.join(config_dir, f"temp_{batch_dir_name}")
    sorcha_out_dir = os.path.join(temp_batch_path, "sorcha_out")

    if os.path.exists(temp_batch_path):
        shutil.rmtree(temp_batch_path)
    os.makedirs(temp_batch_path)

    # sorcha output folder
    os.makedirs(sorcha_out_dir)
        
    try:
        print(f"[{folder_name}] Starting {batch_dir_name}...")

        # generating the orbit and pp files inside the temp sorcha directory
        orbit_file = os.path.join(temp_batch_path, ORBIT_FILENAME)
        pp_file = os.path.join(temp_batch_path, PP_FILENAME)
        complex_file = os.path.join(temp_batch_path, COMPLEX_FILENAME)
        
        # load the precomputed KDE grids and their vmin vmax data
        thin_data = np.load("kde_grid_thin_32.npz")
        thick_data = np.load("kde_grid_thick_32.npz")
        halo_data = np.load("kde_grid_halo_32.npz")

        precomputed_grids = {
            'THIN': {'grid': thin_data['grid'], 'vmin': thin_data['v_min'], 'vmax': thin_data['v_max']},
            'THICK': {'grid': thick_data['grid'], 'vmin': thick_data['v_min'], 'vmax': thick_data['v_max']},
            'HALO': {'grid': halo_data['grid'], 'vmin': halo_data['v_min'], 'vmax': halo_data['v_max']}
        }
        
        total_objects, total_objects_generated = gen_cometary.generate(
            output_dir=temp_batch_path, 
            task_id=config["task_id"],
            alpha=config["alpha"], 
            activity_model=config["activity_model"],
            fractions=config["fractions"],
            precomputed_grids=precomputed_grids
        )

        if total_objects > 0:
            sorcha_command = [
                "python", "run_custom_sorcha.py",
                "-p", pp_file,
                "--ob", orbit_file,
                "--cp", complex_file,
                "-c", INI_FILENAME,
                "--pd", PD_FILENAME,
                "-o", sorcha_out_dir,
                "-t", SORCHA_OUTPUT_PREFIX
            ]
            
            # running sorcha
            sorcha_start = time.time()
            
            #subprocess.run(sorcha_command, check=True, capture_output=True, text=True)

            sorcha_duration = time.time() - sorcha_start
            print(f"   [Performance] Sorcha core execution took {sorcha_duration:.2f} seconds.")
            
            sorcha_csv_original_path = os.path.join(sorcha_out_dir, f"{SORCHA_OUTPUT_PREFIX}.csv")
            sorcha_csv_final_path = os.path.join(temp_batch_path, f"{SORCHA_OUTPUT_PREFIX}.csv")
            
            if os.path.exists(sorcha_csv_original_path):
                shutil.move(sorcha_csv_original_path, sorcha_csv_final_path)
            shutil.rmtree(sorcha_out_dir, ignore_errors=True)
        
        # for debugging potential zero generated object cases
        else:
            print(f"[{folder_name}] 0 valid objects for {batch_dir_name}. Skipping Sorcha execution.")
            with open("empty_batches.log", "a") as empty_log:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                empty_log.write(f"[{timestamp}] ZERO OBJECTS ALERT - {folder_name} / {batch_dir_name}\n")
                empty_log.write(f"Task ID: {config['task_id']}\n")
                empty_log.write(f"Generated (pre-filter): {total_objects_generated} | Valid (post-filter): {total_objects}\n")
                empty_log.write(f"Parameters: Alpha={config['alpha']}, Model={config['activity_model']}\n")
                empty_log.write(f"Fractions: {config['fractions']}\n")
                empty_log.write("-" * 50 + "\n")

        # create the log file
        batch_runtime = time.time() - batch_start_time
        summarize(config, temp_batch_path, batch_dir_name, total_objects, total_objects_generated, batch_runtime)
        
        # temp folder stays in the archive
        os.rename(temp_batch_path, batch_path)

        print(f"[{folder_name}] Successfully finished {batch_dir_name}!")
        
    except subprocess.CalledProcessError as e:
        # sorcha crashed :(
        print(f"!!! [{folder_name}] Sorcha crashed on {batch_dir_name} !!!")
        print(f"Error output: {e.stderr}")
        
        # rescue non-empty sorcha .err files if they exist
        if os.path.exists(sorcha_out_dir):
            for file in os.listdir(sorcha_out_dir):
                if file.endswith(".err"):
                    err_file_path = os.path.join(sorcha_out_dir, file)
                    if os.path.getsize(err_file_path) > 0:
                        shutil.copy(
                            err_file_path, 
                            os.path.join(config_dir, f"CRASH_{batch_dir_name}_{file}")
                        )
        
        shutil.rmtree(temp_batch_path, ignore_errors=True)
        
    except Exception as e:
        # something else crashed ?
        print(f"!!! [{folder_name}] Python error on {batch_dir_name} !!!")
        error_trace = traceback.format_exc()
        print(error_trace)
        
        # write to the main error log in the root directory
        with open("worker_errors.log", "a") as err_file:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            err_file.write(f"[{timestamp}] CRASH IN {folder_name} - {batch_dir_name}\n")
            err_file.write(error_trace)
            err_file.write("-" * 50 + "\n")

        shutil.rmtree(temp_batch_path, ignore_errors=True)

        # wait a bit before continuing
        time.sleep(10)

def summarize(config, temp_dir, batch_dir_name, total_objects, total_objects_generated, batch_runtime):
    """Reads Sorcha output and writes the summary log."""
    output_csv = os.path.join(temp_dir, f"{SORCHA_OUTPUT_PREFIX}.csv")
    total_observations = 0
    pop_detections = {'THIN': 0, 'THICK': 0, 'HALO': 0}
    unique_objects = set()
    
    if os.path.exists(output_csv):
        with open(output_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_observations += 1
                obj_id = row['ObjID']
                
                unique_objects.add(obj_id)
                    
    num_unique = len(unique_objects)

    # detections by parent population
    for obj_id in unique_objects:
        if '_THIN_' in obj_id: pop_detections['THIN'] += 1
        elif '_THICK_' in obj_id: pop_detections['THICK'] += 1
        elif '_HALO_' in obj_id: pop_detections['HALO'] += 1
    
    # detected vs generated
    raw_efficiency = num_unique / total_objects_generated if total_objects_generated > 0 else 0

    # detected vs filtered
    net_efficiency = num_unique / total_objects if total_objects > 0 else 0

    avg_obs = total_observations / num_unique if num_unique > 0 else 0

    # write a human-readable log
    summary_path = os.path.join(temp_dir, f"summary_{batch_dir_name}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"--- BATCH SUMMARY: {batch_dir_name} ---\n")
        f.write(f"Configuration: {config['folder_name']}\n")
        f.write(f"SFD Slope: {config['alpha']}\n")
        f.write(f"Activity Model: {config['activity_model']}\n")
        f.write(f"Runtime: {batch_runtime} seconds\n")
        f.write(f"-----------------------------------\n")
        f.write(f"Total Objects Generated: {total_objects_generated}\n")
        f.write(f"Total Objects After Filtering: {total_objects}\n")
        f.write(f"Total Observations (Rows): {total_observations}\n")
        f.write(f"Unique Objects Detected: {num_unique}\n")
        f.write(f"Average Obs. per Object: {avg_obs:.2f}\n")
        f.write(f"-----------------------------------\n")
        f.write(f"Net Efficiency (Detected / Filtered): {net_efficiency:.2f}\n")
        f.write(f"Raw Efficiency (Detected / Generated): {raw_efficiency:.2f}\n")
        f.write(f"-----------------------------------\n")
        f.write(f"Origin Fractions -> Thin: {config['fractions']['THIN']:.2f} | Thick: {config['fractions']['THICK']:.2f} | Halo: {config['fractions']['HALO']:.2f}\n")
        f.write(f"Thin Disk Detections: {pop_detections['THIN']}\n")
        f.write(f"Thick Disk Detections: {pop_detections['THICK']}\n")
        f.write(f"Halo Detections: {pop_detections['HALO']}\n")
                
        if num_unique == 0:
            f.write(f"NO DETECTIONS IN BATCH")