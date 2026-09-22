import numpy as np

from atmospheric_pde.advection import analytical_advection

def test_analytical_advection():
    x = np.linspace(0, 10, 101)
    
    centre = 2.0
    width = 0.5
    amplitude = 1.0
    
    u = 1.0
    t = 3.0
    
    q_0 = analytical_advection(x, u, t, centre, width, amplitude)
    
    centre_new = centre + u * t
    
    centre_index = np.argmin(np.abs(x - centre_new))
    
    assert np.isclose(q_0[centre_index], amplitude)
    
    
