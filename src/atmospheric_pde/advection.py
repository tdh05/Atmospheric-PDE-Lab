import numpy as np 

from atmospheric_pde.initial_conditions import gaussian

def analytical_advection(x : np.ndarray, u : float, t : float, centre : float, width : float, amplitude : float) -> np.ndarray :
    
    x_departure = x - u * t
    
    return gaussian(x_departure, centre, width, amplitude)


    
    