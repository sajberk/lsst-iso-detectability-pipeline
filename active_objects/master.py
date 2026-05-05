import multiprocessing as mp
import os
import csv
import numpy as np
from config import RUN_CONFIGS, STARTING_CORE, MAX_CORES, N_KINEMATICS, BASE_ARCHIVE_DIR
from worker import run_single_batch
import build_kdes

DEDICATED_CORES = list(range(STARTING_CORE, STARTING_CORE + MAX_CORES))

def pin_worker(cores):
    """Binds a worker process to a specific core on process creation."""
    # get the unique worker id
    worker_id = mp.current_process()._identity[0] 
    # calculate its core number with modulo to handle overflow
    worker_core = cores[(worker_id - 1) % len(cores)] 
    
    os.sched_setaffinity(0, {worker_core})
    print(f"Worker {worker_id} (PID {os.getpid()}) pinned to Core {worker_core}")

if __name__ == '__main__':
    build_kdes.build_all_kdes()

    print(f"Generating {N_KINEMATICS} different kinematics...")
    pop_fractions = np.random.dirichlet(alpha=[1, 1, 1], size=N_KINEMATICS)

    # saving the fractions just in case
    kinematics_record_path = os.path.join(BASE_ARCHIVE_DIR, "kinematics_master_record.csv")
    with open(kinematics_record_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['kin_id', 'THIN', 'THICK', 'HALO'])
        for i, fracs in enumerate(pop_fractions):
            writer.writerow([i, fracs[0], fracs[1], fracs[2]])
    print(f"Kinematics splits saved to {kinematics_record_path}")

    print(f"Assembling the final task list...")
    final_tasks = []
    for config in RUN_CONFIGS:
        kin_id = config["kin_id"]
        fracs = pop_fractions[kin_id]
        
        task = config.copy()

        task["fractions"] = {
            'THIN': fracs[0],
            'THICK': fracs[1],
            'HALO': fracs[2]
        }

        final_tasks.append(task)

    print(f"Starting pipeline on {MAX_CORES} cores with {len(final_tasks)} total tasks...")
    
    with mp.Pool(MAX_CORES, initializer=pin_worker, initargs=(DEDICATED_CORES,), maxtasksperchild=1) as pool:
        # chunksize=1 ensures tasks are grabbed one-by-one with zero idle time
        for _ in pool.imap_unordered(run_single_batch, final_tasks, chunksize=1):
            pass 
        
    print("All tasks completed successfully!")