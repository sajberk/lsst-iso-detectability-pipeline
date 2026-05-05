import numpy as np

# Solar motion relative to LSR
U_sun, V_sun, W_sun = 10.00, 5.25, 7.17

# (Sigma_U, Sigma_V, Sigma_W, V_lag)
pops_params = {
    'THIN':  (35, 20, 16, -15),
    'THICK': (67, 38, 35, -46),
    'HALO':  (160, 90, 90, -220)
}

# helper function
def calc_probability(U, V, W, params):
    """Calculates the probability that a star belongs to a specific population"""
    sig_u, sig_v, sig_w, v_asym = params
    
    k = 1.0 / ((2 * np.pi)**1.5 * sig_u * sig_v * sig_w)
    
    exponent = -0.5 * ( (U**2 / sig_u**2) + 
                        ((V - v_asym)**2 / sig_v**2) + 
                        (W**2 / sig_w**2) )
    
    return k * np.exp(exponent)

def split_stars_by_population(stellar_velocities):
    X_thin = 0.94
    X_thick = 0.06
    X_halo = 0.0015
    
    # Convert heliocentric velocities (m/s) to LSR (km/s) for selection
    U_helio, V_helio, W_helio = stellar_velocities
    u_lsr_km = (U_helio / 1000.0) + U_sun
    v_lsr_km = (V_helio / 1000.0) + V_sun
    w_lsr_km = (W_helio / 1000.0) + W_sun

    # Calculate unweighted probabilities (f)
    f_thin  = calc_probability(u_lsr_km, v_lsr_km, w_lsr_km, pops_params['THIN'])
    f_thick = calc_probability(u_lsr_km, v_lsr_km, w_lsr_km, pops_params['THICK'])
    f_halo  = calc_probability(u_lsr_km, v_lsr_km, w_lsr_km, pops_params['HALO'])

    # Weight the probabilities by the local fraction X
    prob_thin_weighted  = X_thin * f_thin
    prob_thick_weighted = X_thick * f_thick
    prob_halo_weighted  = X_halo * f_halo

    TD_D_ratio = prob_thick_weighted / prob_thin_weighted
    TD_H_ratio = prob_thick_weighted / prob_halo_weighted

    # THIN DISK SELECTION
    idx_thin_pure  = np.where((TD_D_ratio <= 0.1) & (prob_thin_weighted > prob_halo_weighted))[0]

    # THICK DISK SELECTION
    idx_thick_pure = np.where((TD_D_ratio >= 10.0) & (TD_H_ratio >= 1.0))[0]

    # HALO SELECTION
    idx_halo_pure  = np.where((prob_halo_weighted > prob_thin_weighted) & 
                            (prob_halo_weighted > prob_thick_weighted))[0]

    final_samples = {}
    groups = [("THIN", idx_thin_pure), ("THICK", idx_thick_pure), ("HALO", idx_halo_pure)]

    for name, indices in groups:
        final_samples[name] = (U_helio[indices], V_helio[indices], W_helio[indices])

    return final_samples