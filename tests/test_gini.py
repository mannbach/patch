"""Tests the computation of the Gini coefficient from degree distributions.
"""
import numpy as np
from patch.statistics import compute_gini

def test_compute_gini_uniform():
    """Test Gini coefficient for uniform distribution.
    """
    degrees = np.array([10, 10, 10, 10])
    result = compute_gini(degrees)
    assert result == 0.0

def test_compute_gini_increasing():
    """For an increasing sequence, the gini coefficient is expected to be around 0.26667."""
    degrees = np.array([1, 2, 3, 4, 5])
    result = compute_gini(degrees)
    expected = 0.2666666666666667  # computed manually
    assert np.isclose(result, expected, atol=1e-6)

def test_compute_gini_list():
    """Passing a list instead of a numpy array should work
    as np.sort and np.cumsum can handle lists."""
    degrees = [5, 4, 3, 2, 1]
    result = compute_gini(degrees)
    # Sorted list becomes [1,2,3,4,5], same as test_compute_gini_increasing.
    expected = 0.2666666666666667
    assert np.isclose(result, expected, atol=1e-6)
