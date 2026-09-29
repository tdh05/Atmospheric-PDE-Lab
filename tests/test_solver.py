import numpy as np
import pytest

from atmospheric_pde.advection import upwind_step
from atmospheric_pde.solver import solve_upwind

def test_zero_steps():
    """Testing if using n_steps = 0 correctly returns the initial conditions"""
    
    q_0 = np.array([0.0, 1.0, 2.0, 3.0])
    
    u = 1.0
    dt = 0.5
    dx = 1.0
    n_steps = 0
    
    output = solve_upwind(q_0, u, dt, dx, n_steps)    
    
    assert np.allclose(output, q_0)
    
def test_multistep():
    """Testing if using multiple steps matches the result for repeated single steps"""
    
    q_0 = np.array([0.0, 1.0, 2.0, 3.0])
    
    u = 1.0
    dt = 0.5
    dx = 1.0
    
    output = solve_upwind(q_0, u, dt, dx, n_steps = 2)  
    
    expected = upwind_step(q_0, u, dt, dx)
    expected = upwind_step(expected, u, dt, dx)
    
    assert np.allclose(output, expected)
    

def test_multistep_conservation():
    q_0 = np.array([0.0, 1.0, 2.0, 3.0])
    
    u = 1.0
    dt = 0.5
    dx = 1.0
    
    output = solve_upwind(q_0, u, dt, dx, n_steps = 10)
    
    initial_mass = np.sum(q_0) * dx
    final_mass = np.sum(output) * dx
    
    assert np.isclose(final_mass, initial_mass)
    
def test_solver_input_validation():
    """Testing the input validation steps implemented in the upwind solver function"""
    
    q_0 = np.array([0.0, 1.0, 2.0, 3.0])
    
    with pytest.raises(ValueError):
        solve_upwind(q_0, u = 1.0, dt = 0.0, dx = 1.0, n_steps = 10)
        
    with pytest.raises(ValueError):
        solve_upwind(q_0, u = 1.0, dt = 0.5, dx = 0.0, n_steps = 10)
        
    with pytest.raises(ValueError):
        solve_upwind(q_0, u = 1.0, dt = 0.5, dx = 1.0, n_steps = -1)
    