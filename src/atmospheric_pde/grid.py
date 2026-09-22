import numpy as np

def create_grid(L: float, N: float) -> np.ndarray:
    ''''Create a uniform one-dimensional grid'''
    
    return np.linspace(0.0, L , N)
