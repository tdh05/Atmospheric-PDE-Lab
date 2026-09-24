import numpy as np

from atmospheric_pde.advection import upwind_step

def solve_upwind(q_0 : np.ndarray, u : float, dt : float, dx : float, n_steps : int) -> np.ndarray:
    
    if dt <= 0 :
        raise ValueError(f"Entered dt value : {dt}, must be positive")
    if dx <= 0 :
        raise ValueError(f"Entered dx value : {dx}, must be positive")
    if n_steps < 0 :
        raise ValueError(f"Entered n_steps value : {n_steps}, must be greater than or equal to 0")
    
    q = np.copy(q_0)
    
    for n in range(n_steps):
        q = upwind_step(q, u, dt, dx)
        
    return q