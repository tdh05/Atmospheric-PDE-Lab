import numpy as np 


def gaussian(x : np.ndarray, centre : float, width : float, amplitude : float = 1.0) -> np.ndarray:
    """Return a Gaussian function of q(x, 0) for every grid point (tracer distribution)"""
    
    return amplitude * np.exp(- ((x - centre)**2) / (2 * width**2))
    