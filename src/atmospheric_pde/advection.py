import numpy as np 

from atmospheric_pde.initial_conditions import gaussian

def analytical_advection(x : np.ndarray, u : float, t : float, centre : float, width : float, amplitude : float) -> np.ndarray :
    
    x_departure = x - u * t
    
    return gaussian(x_departure, centre, width, amplitude)


def upwind_step(q : np.ndarray, u : float, dt : float, dx : float) -> np.ndarray:
   
    if u <= 0:
        raise ValueError(f"The provided value of u : {u} cannot be less than or equal to 0")
    
    C = (u * dt) / (dx)
    
    q_new = np.empty_like(q)
    
    for i in range(len(q)):
        q_new[i] = q[i] - C * (q[i] - q[i-1])
        
    return q_new


def analytical_periodic_advection(x : np.ndarray, u : float, t : float, centre : float, width : float, amplitude : float, domain_length: float, K : float) -> np.ndarray :
    
    q_exact = np.zeros_like(x)
    
    for K in range(-K, K + 1):
        x_departure = x - (u * t) + (K * domain_length)
        
        q_exact += gaussian(x_departure, centre, width, amplitude)
        
    return q_exact 
    
        
    
    
    
    
    