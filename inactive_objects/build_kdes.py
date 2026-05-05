import numpy as np
import os
from sklearn.neighbors import KernelDensity
from gaia_loader import fetch_gaia_velocities
from kinematics_splitter import split_stars_by_population

SPEED_RESOLUTION = 50
ANGLE_RESOLUTION = 60

def build_static_kde(U, V, W, v_arr, l_arr, b_arr):
    values = np.vstack([U, V, W])
    n_samples = values.shape[1]
    
    # Scott's Rule for bandwidth
    scott_bw = n_samples**(-1. / (3 + 4)) * np.mean(np.std(values, axis=1))
    kde = KernelDensity(bandwidth=scott_bw, algorithm='kd_tree', rtol=0.01, leaf_size=100)
    kde.fit(values.T)
    
    # Generate 3D grid
    ind = np.mgrid[0:len(v_arr), 0:len(l_arr), 0:len(b_arr)]
    v_base = v_arr[ind[0]]
    l_base = l_arr[ind[1]]
    b_base = b_arr[ind[2]]
    
    # Convert spherical to cartesian for evaluation
    vx_base = -v_base * np.cos(b_base) * np.cos(l_base)
    vy_base = -v_base * np.cos(b_base) * np.sin(l_base)
    vz_base = -v_base * np.sin(b_base)
    
    eval_points = np.column_stack([vx_base.ravel(), vy_base.ravel(), vz_base.ravel()])
    log_pdf = kde.score_samples(eval_points)
    pdf_values = np.exp(log_pdf)
    
    # Convert back to spherical probability density: p(v, l, b)
    p_vlb = pdf_values.reshape(v_base.shape) * (v_base**2) * np.cos(b_base)
    return p_vlb

def build_all_kdes():
    """Function called by master.py to ensure KDE files exist."""

    # Check if files already exist to save time on restarts
    required_files = ["kde_grid_thin.npz", "kde_grid_thick.npz", "kde_grid_halo.npz"]
    if all(os.path.exists(f) for f in required_files):
        print("KDE grids already exist. Skipping build phase.")
        return
    
    print("--- Starting KDE Build Phase ---")

    # 1. Setup the shared angle grids (these remain static)
    l_arr = np.linspace(0, 2*np.pi, ANGLE_RESOLUTION)
    u_arr = np.linspace(-1, 1, ANGLE_RESOLUTION//2)
    b_arr = np.arcsin(u_arr)
    
    # ---------------------------------------------------------
    # PHASE 1: THIN DISK (100 pc limit)
    # ---------------------------------------------------------
    print("Fetching 100 pc volume for Thin Disk...")
    U_100, V_100, W_100 = fetch_gaia_velocities(p_min=10)
    pops_100 = split_stars_by_population((U_100, V_100, W_100))
    
    U_thin, V_thin, W_thin = pops_100['THIN']
    speeds_thin = np.sqrt(U_thin**2 + V_thin**2 + W_thin**2)
    vmin_thin = np.min(speeds_thin) * 0.95
    vmax_thin = np.max(speeds_thin) * 1.05
    v_arr_thin = np.linspace(vmin_thin, vmax_thin, SPEED_RESOLUTION)

    print(f"Building grid for THIN disk ({len(U_thin)} stars) | Bounds: {vmin_thin:.1f} to {vmax_thin:.1f} m/s")
    p_vlb_thin = build_static_kde(U_thin, V_thin, W_thin, v_arr_thin, l_arr, b_arr)
    np.savez("kde_grid_thin.npz", grid=p_vlb_thin, v_min=vmin_thin, v_max=vmax_thin)

    # ---------------------------------------------------------
    # PHASE 2: THICK & HALO DISKS (200 pc limit)
    # ---------------------------------------------------------
    print("\nFetching 200 pc volume for Thick & Halo Disks...")
    U_200, V_200, W_200 = fetch_gaia_velocities(p_min=5)
    pops_200 = split_stars_by_population((U_200, V_200, W_200))
    
    # THICK DISK
    U_thick, V_thick, W_thick = pops_200['THICK']
    speeds_thick = np.sqrt(U_thick**2 + V_thick**2 + W_thick**2)
    vmin_thick = np.min(speeds_thick) * 0.95
    vmax_thick = np.max(speeds_thick) * 1.05
    v_arr_thick = np.linspace(vmin_thick, vmax_thick, SPEED_RESOLUTION)

    print(f"Building grid for THICK disk ({len(U_thick)} stars) | Bounds: {vmin_thick:.1f} to {vmax_thick:.1f} m/s")
    p_vlb_thick = build_static_kde(U_thick, V_thick, W_thick, v_arr_thick, l_arr, b_arr)
    np.savez("kde_grid_thick.npz", grid=p_vlb_thick, v_min=vmin_thick, v_max=vmax_thick)
    
    # HALO DISK
    U_halo, V_halo, W_halo = pops_200['HALO']
    speeds_halo = np.sqrt(U_halo**2 + V_halo**2 + W_halo**2)
    vmin_halo = np.min(speeds_halo) * 0.95
    vmax_halo = np.max(speeds_halo) * 1.05
    v_arr_halo = np.linspace(vmin_halo, vmax_halo, SPEED_RESOLUTION)

    print(f"Building grid for HALO disk ({len(U_halo)} stars) | Bounds: {vmin_halo:.1f} to {vmax_halo:.1f} m/s")
    p_vlb_halo = build_static_kde(U_halo, V_halo, W_halo, v_arr_halo, l_arr, b_arr)
    np.savez("kde_grid_halo.npz", grid=p_vlb_halo, v_min=vmin_halo, v_max=vmax_halo)

    print("\nAll KDE grids successfully built and saved with dynamic bounds!")

if __name__ == "__main__":
    build_all_kdes()