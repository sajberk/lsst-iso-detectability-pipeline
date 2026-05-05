import os
import numpy as np
import pandas as pd
from scipy.stats import lognorm, norm, gaussian_kde

from config import ORBIT_FILENAME, PP_FILENAME, COMPLEX_FILENAME, EPOCH, TIME_OF_SIMULATION
from config import N0, D_REF, D_STEPS
from misc_functions import filter_and_convert
from synthetic_population_scikit import synthetic_population
from utils import max_hc_distance_asteroid, absolute_magnitude_asteroid


mu = 1.3271244e+20
au = 149597870700.0
COMET_ALBEDO = 0.04

COOK_MODELS = {
    "Kresak": {"b1": -0.20, "b2": 2.10},
    "Bailey & Stagg": {"b1": -0.17, "b2": 1.90},
    "Weissman": {"b1": -0.13, "b2": 1.86},
    "Sosa & Fernandez": {"b1": -0.13, "b2": 1.20},
    }

USE_LOG_NORMAL = True
JPL_CACHE_FILE = "jpl_empirical_model.npz"
if os.path.exists(JPL_CACHE_FILE):
    jpl_data = np.load(JPL_CACHE_FILE)
    jpl_n_kde = gaussian_kde(jpl_data['n_values'], bw_method=0.15)
else:
    jpl_data, jpl_n_kde = None, None


def generate(output_dir, task_id, alpha, activity_model, fractions, precomputed_grids):
    # initializing these list here and then will them in the for loops
    df_orbits_list = []
    df_pp_list = []
    df_complex_list = []
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
            # RM calculations here
            if activity_model in COOK_MODELS:
                b1 = COOK_MODELS[activity_model]["b1"]
                b2 = COOK_MODELS[activity_model]["b2"]
                r_km_max = (d[-1] / 2.0) * 1e-3
                H_c_min = (np.log10(2 * r_km_max) - b2) / b1

                RM_comet = 10 ** ((24.5 - H_c_min) / 17.5)

                # since n calcs strongly penalize distant comets
                RM_asteroid = max_hc_distance_asteroid(d[-1], COMET_ALBEDO, 24.5)
                RM = max(RM_comet, RM_asteroid)

            elif activity_model == "JPL":
                M2_max = absolute_magnitude_asteroid(d[-1], COMET_ALBEDO)
                if USE_LOG_NORMAL:
                    delta_M_99 = lognorm.ppf(0.99, jpl_data['ln_shape'], loc=jpl_data['ln_loc'], scale=jpl_data['ln_scale'])
                else:
                    delta_M_99 = norm.ppf(0.99, jpl_data['norm_mu'], jpl_data['norm_std'])
                H_c_min = M2_max - delta_M_99

                RM_comet = 10 ** ((24.5 - H_c_min) / 17.5)
                RM_asteroid = max_hc_distance_asteroid(d[-1], COMET_ALBEDO, 24.5)
                RM = max(RM_comet, RM_asteroid)

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

            if activity_model in COOK_MODELS:
                r_km = (D_m / 2.0) * 1e-3
                H_c = (np.log10(2 * r_km) - b2) / b1
                hc_max_comet_arr = 10 ** ((24.5 - H_c) / 17.5)

            elif activity_model == "JPL":
                M2_arr = absolute_magnitude_asteroid(D_m, COMET_ALBEDO)
                if USE_LOG_NORMAL:
                    delta_M_arr = lognorm.rvs(jpl_data['ln_shape'], loc=jpl_data['ln_loc'], scale=jpl_data['ln_scale'], size=num_objects_old)
                else:
                    delta_M_arr = np.random.normal(jpl_data['norm_mu'], jpl_data['norm_std'], size=num_objects_old)
                delta_M_arr = np.maximum(delta_M_arr, 0.1)
                H_c = M2_arr - delta_M_arr
                hc_max_comet_arr = 10 ** ((24.5 - H_c) / 17.5)
            

            filtered_objects = filter_and_convert(q, e, inc, Omega, omega, f, D_m, COMET_ALBEDO, H_c, is_comet=True, hc_max_comet=hc_max_comet_arr)
            q_out, e_out, inc_out, Omega_out, omega_out, tp_out, D_out, H_out, H_c_out = filtered_objects
            num_objects = len(q_out)


            # ovde racunamo jos samo n za cpp.txt
            if activity_model in COOK_MODELS:
                n_index_out = [4.0] * num_objects
        
            elif activity_model == "JPL":
                n_index_out = jpl_n_kde.resample(num_objects)[0]


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

            df_complex_batch = pd.DataFrame({
                'H_c': H_c_out, 
                'n_index': n_index_out
            })

            obj_ids = [f"ISO_{task_id:04d}_{name}_{this_pop_obj_counter + i:07d}" for i in range(num_objects)]

            df_orbits_batch.insert(0, 'ObjID', obj_ids)
            df_pp_batch.insert(0, 'ObjID', obj_ids)
            df_complex_batch.insert(0, 'ObjID', obj_ids)

            df_orbits_list.append(df_orbits_batch)
            df_pp_list.append(df_pp_batch)
            df_complex_list.append(df_complex_batch)

            this_pop_obj_counter += num_objects

    
    # combine both batches into a single dataframe
    total_valid_objects = sum(len(df) for df in df_orbits_list)

    if total_valid_objects > 0:
        df_orbits = pd.concat(df_orbits_list, ignore_index=True)
        df_pp = pd.concat(df_pp_list, ignore_index=True)
        df_complex = pd.concat(df_complex_list, ignore_index=True)

        # clean up nans
        valid_indices = df_orbits.dropna().index
        df_orbits = df_orbits.loc[valid_indices].reset_index(drop=True)
        df_pp = df_pp.loc[valid_indices].reset_index(drop=True)
        df_complex = df_complex.loc[valid_indices].reset_index(drop=True)
    else:
        # if we get 0 valid objects
        df_orbits = pd.DataFrame(columns=['ObjID', 'FORMAT', 'q', 'e', 'inc', 'node', 'argPeri', 't_p_MJD_TDB', 'epochMJD_TDB'])
        df_pp = pd.DataFrame(columns=['ObjID', 'H_r', 'u-r', 'g-r', 'i-r', 'z-r', 'y-r', 'GS'])
        df_complex = pd.DataFrame(columns=['ObjID', 'H_c', 'n_index'])

    # finally, save everything
    orbit_path = os.path.join(output_dir, ORBIT_FILENAME)
    pp_path = os.path.join(output_dir, PP_FILENAME)
    complex_path = os.path.join(output_dir, COMPLEX_FILENAME)

    df_orbits.to_csv(orbit_path, index=False, sep=',')
    df_pp.to_csv(pp_path, index=False, sep=',')
    df_complex.to_csv(complex_path, index=False, sep=',')


    return len(df_orbits), total_num_objects_old