import os
import numpy as np
import pandas as pd

from config import ORBIT_FILENAME, PP_FILENAME, EPOCH, TIME_OF_SIMULATION
from config import N0, D_REF, D_STEPS
from misc_functions import filter_and_convert
from synthetic_population_scikit import synthetic_population
from utils import max_hc_distance_asteroid

mu = 1.3271244e+20
au = 149597870700.0

def generate(output_dir, task_id, alpha, albedo, fractions, precomputed_grids):
    # initializing these list here and then will them in the for loops
    df_orbits_list = []
    df_pp_list = []
    total_num_objects_old = 0

    for name, pop_data in precomputed_grids.items():
        pop_fraction = fractions[name]
        adjusted_n0 = N0 * pop_fraction

        # for sequential ids within this population
        this_pop_obj_counter = 0

        pop_vmin = pop_data['vmin']
        pop_vmax = pop_data['vmax']
        pop_grid = pop_data['grid']

        # run the generation for each diameter range
        for d in D_STEPS:
            # determine the RM dynamically based on the current range's max diameter
            if albedo == "trojan":
                safe_max_albedo = 0.2
                RM = max_hc_distance_asteroid(d[-1], safe_max_albedo, 24.5)
            else:
                RM = max_hc_distance_asteroid(d[-1], albedo, 24.5)

            results = synthetic_population(
                TIME_OF_SIMULATION, rm=RM, n0=adjusted_n0, v_min=pop_vmin, v_max=pop_vmax,
                u_Sun=0, v_Sun=0, w_Sun=0, 
                sigma_vx=0, sigma_vy=0, sigma_vz=0,
                vd=0, va=0, R_reff=696340000.,
                d_ref=D_REF, d=d, alpha=[alpha],
                precomputed_p_vlb=pop_grid
            )

            if results is None:
                print(f"Generated 0 objects for D={d} in {name}. Skipping...")
                continue

            q, e, f, inc, Omega, omega, D_m = results

            num_objects_old = len(q)
            total_num_objects_old += num_objects_old

            print(f"Generated {num_objects_old} objects for D={d} in {name}. Filtering...")

            # assign albedos to the generated objects
            if albedo == "trojan":
                from wise_albedo_generator import wise_albedo_diameter
                object_albedos = wise_albedo_diameter.get_albedos(num_objects_old)
            else:
                # constant float albedo
                object_albedos = albedo
            
            filtered_objects = filter_and_convert(q, e, inc, Omega, omega, f, D_m, object_albedos)
            q_out, e_out, inc_out, Omega_out, omega_out, tp_out, D_out, H_out = filtered_objects

            num_objects = len(q_out)

            # create small batch dfs without ids, as we will add these later when we merge them all
            df_orbits_batch = pd.DataFrame({
                'FORMAT': ['COM'] * num_objects,
                'q': q_out, 'e': e_out, 'inc': inc_out, 'node': Omega_out, 'argPeri': omega_out,
                't_p_MJD_TDB': tp_out, 'epochMJD_TDB': [EPOCH] * num_objects
            })

            df_pp_batch = pd.DataFrame({
                'H_r': H_out, 'u-r': [2.31] * num_objects, 'g-r': [0.73] * num_objects, 
                'i-r': [-0.33] * num_objects, 'z-r': [-0.5] * num_objects, 
                'y-r': [-0.68] * num_objects, 'GS': [0.15] * num_objects,
            })

            obj_ids = [f"ISO_{task_id:04d}_{name}_{this_pop_obj_counter + i:07d}" for i in range(num_objects)]
            
            df_orbits_batch.insert(0, 'ObjID', obj_ids)
            df_pp_batch.insert(0, 'ObjID', obj_ids)

            df_orbits_list.append(df_orbits_batch)
            df_pp_list.append(df_pp_batch)

            this_pop_obj_counter += num_objects

    # combine both batches into a single dataframe
    total_valid_objects = sum(len(df) for df in df_orbits_list)

    if total_valid_objects > 0:
        df_orbits = pd.concat(df_orbits_list, ignore_index=True)
        df_pp = pd.concat(df_pp_list, ignore_index=True)

        # clean up nans
        valid_indices = df_orbits.dropna().index
        df_orbits = df_orbits.loc[valid_indices].reset_index(drop=True)
        df_pp = df_pp.loc[valid_indices].reset_index(drop=True)
    else:
        # if we get 0 valid objects
        df_orbits = pd.DataFrame(columns=['ObjID', 'FORMAT', 'q', 'e', 'inc', 'node', 'argPeri', 't_p_MJD_TDB', 'epochMJD_TDB'])
        df_pp = pd.DataFrame(columns=['ObjID', 'H_r', 'u-r', 'g-r', 'i-r', 'z-r', 'y-r', 'GS'])

    # finally, save everything
    orbit_path = os.path.join(output_dir, ORBIT_FILENAME)
    pp_path = os.path.join(output_dir, PP_FILENAME)

    df_orbits.to_csv(orbit_path, index=False, sep=',')
    df_pp.to_csv(pp_path, index=False, sep=',')

    return len(df_orbits), total_num_objects_old