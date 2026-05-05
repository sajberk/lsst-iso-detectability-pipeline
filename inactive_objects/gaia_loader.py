import warnings
from astropy.coordinates import SkyCoord
import numpy as np
import astropy.units as u

def fetch_gaia_velocities(p_min=10, limit=None):
    """
    Fetches the master dataset of all valid stars within a given parallax limit.
    Returns the Galactic U, V, W velocity arrays in m/s.
    """
    where_clause = f"""
    WHERE 
        parallax > {p_min}
        AND radial_velocity IS NOT NULL
        AND (parallax / parallax_error) > 10 
        AND visibility_periods_used > 8
        AND astrometric_n_obs_al > 100
        AND ruwe < 1.4
    """
    
    cols = "ra, dec, parallax, pmra, pmdec, radial_velocity"
    results = None
    
    try:
        from astroquery.gaia import Gaia
        a = 5/0
        print("Fetching master Gaia data via Astroquery...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            query = f"SELECT {f'TOP {limit}' if limit else ''} {cols} FROM gaiadr3.gaia_source {where_clause}"
            job = Gaia.launch_job_async(query)
            results = job.get_results()
            
    except Exception as e:
        print(f"Astroquery failed: {e}\nFalling back to AIP Mirror...")
        import pyvo
        tap_service = pyvo.dal.TAPService("https://gaia.aip.de/tap")
        query = f"SELECT {f'TOP {limit}' if limit else ''} {cols} FROM gaiadr3.gaia_source {where_clause}"
        try:
            job = tap_service.submit_job(query)
            job.run()
            results = job.fetch_result().to_table()
        except:
            results = tap_service.run_async(query).to_table()

    print(f"Successfully fetched {len(results)} stars.")

    c = SkyCoord(ra=np.array(results['ra']) * u.deg,
                dec=np.array(results['dec']) * u.deg,
                distance=(1000.0 / np.array(results['parallax'])) * u.pc,
                pm_ra_cosdec=np.array(results['pmra']) * (u.mas / u.yr),
                pm_dec=np.array(results['pmdec']) * (u.mas / u.yr),
                radial_velocity=np.array(results['radial_velocity']) * (u.km / u.s)
    )

    gal = c.galactic
    U = gal.velocity.d_x.to(u.m/u.s).value
    V = gal.velocity.d_y.to(u.m/u.s).value
    W = gal.velocity.d_z.to(u.m/u.s).value
    
    return U, V, W