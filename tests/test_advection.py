import numpy as np
import pytest

from atmospheric_pde.advection import (analytical_advection,
                                       upwind_step)

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
    
def test_upwind_step():
    q = np.ones(5)
    
    u = 1.0
    dt = 0.5
    dx = 1.0
    
    output = upwind_step(q, u, dt, dx)
    
    assert np.allclose(output, q)
    
def test_conservation():
    q = np.array([0.0, 1.0, 2.0, 3.0])
    
    u = 1.0
    dt = 0.5
    dx = 1.0
    
    output = upwind_step(q, u, dt, dx)
    
    initial_mass = np.sum(q) * dx
    new_mass = np.sum(output) * dx
    
    assert np.isclose(new_mass, initial_mass)
    
def test_upwind_positive_velocity():
    q = np.ones(5)
    
    u = -1.0
    dt = 0.5
    dx = 1.0
    
    with pytest.raises(ValueError):
        upwind_step(q, u, dt, dx)

