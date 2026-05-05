import numpy as np
import os
import pandas as pd
from scipy.stats import gaussian_kde
import requests
from utils import true2ecc, ecc2mean, mean2tp, max_hc_distance_asteroid, year2sec, orb2cart, gal2ecl_cart, cart2orb, absolute_magnitude_asteroid, mu, au
from config import TIME_OF_SIMULATION, EPOCH

def filter_and_convert(q, e, inc, Omega, omega, f, D_m, albedo):
    """ Filters out objects that won't be observable and calculates tp and H for sorcha"""

    # fix for wise albedo generator
    albedo_is_array = np.ndim(albedo) > 0
    
    # filtering 
    """
    SELECTION No. 1
    We select only object whose perihelion distance is smaller than hc_max.
    """
    hc_max = max_hc_distance_asteroid(D_m, albedo, 24.5)           

    selection1 = q < hc_max  # This is OK (double checked)

    ecc2 = e[selection1]
    f2 = f[selection1]
    inc2 = inc[selection1]
    Omega2 = Omega[selection1]
    omega2 = omega[selection1]
    q2 = q[selection1]
    D2 = D_m[selection1]
    hc_max2 = hc_max[selection1]
    
    # --- FIX: Slice the albedo array if it exists! ---
    if albedo_is_array:
        albedo2 = albedo[selection1]
    else:
        albedo2 = albedo

    '''
    SELECTION No. 2
    Sada za svaki objekat proveravamo da li je moguca da bude uhvacen. Imamo 3 opcije:
        
    1. Nalazi se unutar svoje observabilne sfere, tj. pocetno helicentricno rastojanje mu je manje od onog na kom moze biti posmatran (OSTAJE)
    2. Pocetno rastojanje mu je vece od onog na kom moze biti posmatran i anomalija mu je pozitivna. Ovo znaci da se udaljava i nece biti posomatran (IZBACUJEMO)
    3. Pocetno rastojanje mu je vece od onog na kom moze biti posmatran i anomalija mu je negativna. Ovo znaci da se priblizava i mozda ce biti posmatran. Proveravamo da li moze
    da stigne i odlucujemo da li ga zadrzavamo.
    '''

    #Equation of hyperbolic orbit:
    #r = a*(1-e*cosh(E)), where E is hyperbolic anomaly
    #
    #From this equation, given eccentricity, semi-major axis and maximum observable heliocentric distance
    #we can calculate critical hyperbolic anomaly (when ISO is exactly at hc_max) 

    Ecr = np.arccosh(1 / ecc2 - hc_max2 / ecc2 / (q2 / (1 - ecc2)))  # This is OK (double checked) this is always positive

    # corresponding critical mean anomaly (maximum where an object of a given size can be observed)
    # from hyperbolic Kepler equation
    # M = e *sinh(E) - E

    M_max = ecc2 * np.sinh(Ecr) - Ecr  # This is OK (double checked) this is always positive

    # We calculate M for every object from the population

    M = np.zeros(len(q2))  # current mean anomaly
    for i in range(len(q2)):
        E = true2ecc(f2[i], ecc2[i])  # This is OK (double checked)
        M[i] = ecc2mean(E, ecc2[i])  # This is OK (double checked)


    """
    We calculate mean motion
    """
    mean_motion = np.sqrt(mu / (np.abs(q2 / (1 - ecc2)) * au) ** 3)

    """
    Finally, we calculate minimum mean anomaly from which an object can reach hc_max during the simulation time
    """
    M_min = -M_max - mean_motion * year2sec(TIME_OF_SIMULATION)  # This is OK (double checked)

    """                        
    If object is outside observable sphere and it is on outgoing branch or orbit (Mean anomaly > M_max) it is excluded
    because it is surely non observable. For objects with positive mean anomaly (those which are on the outgoing branch)
    we take only those which are inside their observable spheres


    For objects on incoming branches we take those which are currently observable but also those whose mean anomaly 
    is larger than M_min. This means that those object will reach their observable sphere during the OIF simualtion time. 
    """

    selection2 = np.logical_and(M > M_min, M < M_max)  # This is OK (double checked)
    e_out = ecc2[selection2]
    f_out = f2[selection2]
    inc_out = inc2[selection2]
    Omega_out = Omega2[selection2]
    omega_out = omega2[selection2]
    q_out = q2[selection2]
    D_out = D2[selection2]

    if albedo_is_array:
        albedo_out = albedo2[selection2]
    else:
        albedo_out = albedo2

    H_out = absolute_magnitude_asteroid(D_out, albedo_out)

    for i in range(len(q_out)):
            x,y,z,vx,vy,vz=orb2cart(omega_out[i], Omega_out[i], inc_out[i], e_out[i], q_out[i]/(1-e_out[i])*au, 0.1, mu) # we put E=0.1 (just not at the pericenter) because it does not impact the conversion of o, O, inc (This is OK - double checked)
            
            x,y,z=gal2ecl_cart(x,y,z) # This is OK (double checked)
            vx,vy,vz=gal2ecl_cart(vx,vy,vz)  # This is OK (double checked)  
            omega_out[i], Omega_out[i], inc_out[i]=cart2orb(x,y,z,vx,vy,vz, mu)[:3] # This is OK (double checked)
        
        
    tp_out=np.zeros(len(q_out))

    for i in range(len(q_out)):

        tp_out[i] = mean2tp(ecc2mean(true2ecc(f_out[i], e_out[i]), e_out[i]), q_out[i] / (1 - e_out[i]), EPOCH)  # This is OK (double checked)


    inc_out = np.rad2deg(inc_out)
    Omega_out = np.rad2deg(Omega_out)
    omega_out = np.rad2deg(omega_out)

    return q_out, e_out, inc_out, Omega_out, omega_out, tp_out, D_out, H_out
