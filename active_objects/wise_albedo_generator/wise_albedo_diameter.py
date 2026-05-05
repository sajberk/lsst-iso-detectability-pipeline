import numpy as np
import pandas as pd
from .auxiliary_functions import imitate_sample

def get_albedos(N):
    '''
    Takes albedo and/or diameter data from WISE and generates a sample of arbitrary
    size that imitates the WISE data
    '''

    n_bins=20 # number of bins for the available data
    file='wise_albedo_generator/neowise_jupiter_trojans.csv'

    H, D, aV = (pd.read_csv(file, usecols=[3,11,13])).T.to_numpy() # za trojance

    aV = aV[aV>0]
    variable=aV

    # -0.999 is a flag for not having albedo
    index = np.argwhere(variable==-0.999)
    variable = np.delete(variable, index)

    synthetic_sample=imitate_sample(variable, n_bins, N)
    
    return synthetic_sample
