import requests
import numpy as np
import pandas as pd
from scipy.stats import lognorm, norm, gaussian_kde

print("Fetching JPL SBDB data for Model 5...")
URL = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
params = {'sb-kind': 'c', 'fields': 'pdes,M1,M2,K1,class'}
response = requests.get(URL, params=params).json()

df = pd.DataFrame(response['data'], columns=response['fields'])
for col in ['M1', 'M2', 'K1']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df[df['class'].isin(['PAR', 'HYP'])]
clean_df = df.dropna(subset=['M1', 'M2', 'K1']).copy()
clean_df['n'] = clean_df['K1'] / 2.5

mask_defaults = np.isclose(clean_df['n'], 4.0, atol=0.05) | np.isclose(clean_df['n'], 10.0, atol=0.05) | np.isclose(clean_df['n'], 2.0, atol=0.05) | np.isclose(clean_df['n'], 1.8, atol=0.05)
filtered_df = clean_df[~mask_defaults]

# 1. Delta M Fits (Saving BOTH)
delta_M = (filtered_df['M2'] - filtered_df['M1']).values
delta_M = delta_M[delta_M > 0] 

shape, loc, scale = lognorm.fit(delta_M, floc=0)
mu_n, std_n = norm.fit(delta_M)

# 2. Activity Index 'n' Fit (1D KDE)
n_values = filtered_df['n'].values
n_kde = gaussian_kde(n_values, bw_method=0.15)

np.savez("jpl_empirical_model.npz", 
         ln_shape=shape, ln_loc=loc, ln_scale=scale, 
         norm_mu=mu_n, norm_std=std_n, 
         n_values=n_values)

print("Saved jpl_empirical_model.npz successfully!")