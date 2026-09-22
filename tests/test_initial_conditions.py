import numpy as np

from atmospheric_pde.initial_conditions import gaussian

def test_gaussian_shape():
    x = np.array([0.0, 1.0, 2.0])
    
    q = gaussian(x, 1.0, 1.0)
    
    assert np.isclose(q[1], 1.0)
    assert isinstance(q, np.ndarray)