import numpy as np 

from atmospheric_pde.grid import create_grid

L = 10.0
N = 11

def test_create_grid():
    
    x = create_grid(L, N)
    
    spacing = np.diff(x)
    
    assert np.allclose(spacing, spacing[0])