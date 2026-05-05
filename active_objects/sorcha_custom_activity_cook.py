import numpy as np
import pandas as pd
from typing import List
from sorcha.activity.base_activity import AbstractCometaryActivity

class Cook2016Activity(AbstractCometaryActivity):
    """
    Custom Sorcha cometary activity model based strictly on Cook et al. (2016).
    
    Methodology:
    Calculates the apparent magnitude of the active coma (Equation 5) and compares 
    it directly to the bare-nucleus asteroid magnitude. The final observable magnitude 
    is defined as "whichever is brighter" between the two. 
    """

    def __init__(self, required_column_names: List[str] = ['H_c', 'n_index']):
        super().__init__(required_column_names)

    @staticmethod
    def name_id() -> str:
        return "sorcha_custom_activity_cook"

    def compute(
        self,
        df: pd.DataFrame,
        observing_filters: List[str],
        rho: List[float],
        delta: List[float],
        alpha: List[float],
    ) -> pd.DataFrame:
        
        self._validate_column_names(df)
        alpha_deg = df['phase_deg']
        alpha_deg = np.array(alpha_deg)
        alpha = np.deg2rad(alpha_deg)
        # Phase Function (\gamma) Calculation (Bowell et al. with G = 0.15)
        G = 0.15
        phi_1 = np.exp(-3.33 * np.tan(alpha / 2)**0.63)
        phi_2 = np.exp(-1.87 * np.tan(alpha / 2)**1.22)
        gamma = (1 - G) * phi_1 + G * phi_2
        phase_term = -2.5 * np.log10(gamma)

        H_c = df['H_c'].values
        n = df['n_index'].values

        # Cook et al. (2016) Equation 5: 
        # V = H_c + 2.5 * [ (n/2)*log10(rho^2) + log10(delta^2) ]
        # This translates to the standard coma equation, omitting a phase angle term.
        m_comet = H_c + 5.0 * np.log10(delta) + 2.5 * n * np.log10(rho) + phase_term

        # "The apparent magnitude is then taken to be whichever is brighter 
        # out of the asteroid and comet apparent magnitudes"
        # Sorcha's 'trailedSourceMagTrue' provides the baseline asteroid magnitude.
        m_ast = df['trailedSourceMagTrue'].values
        
        # Apply the piecewise 'whichever is brighter' logic
        df["trailedSourceMagTrue"] = np.minimum(m_comet, m_ast)

        return df

    def maxBrightness(
        self,
        df: pd.DataFrame,
        observing_filters: List[str],
        q: List[float],
        delta: List[float],
        alpha: List[float],
    ) -> float:
        """
        Calculates the theoretical maximum brightness to allow Sorcha to cull 
        undetectable objects early, avoiding false-culling from poor instantaneous geometry.
        """
        self._validate_column_names(df)

        H_c = df['H_c'].values
        n = df['n_index'].values

        q_arr = np.array(q)
        
        # Absolute optimal Earth distance (closest theoretical approach)
        delta_min = np.maximum(np.abs(q_arr - 1.0), 0.05)

        # Maximum brightness of the coma component at optimal orbital geometry
        m_comet_max = H_c + 5.0 * np.log10(delta_min) + 2.5 * n * np.log10(q_arr)

        # Safely catch against the instantaneous bare rock magnitude to prevent 
        # culling objects that are completely inactive but close to Earth.
        brightestMag = np.minimum(m_comet_max, df['trailedSourceMagTrue'].values)

        return brightestMag