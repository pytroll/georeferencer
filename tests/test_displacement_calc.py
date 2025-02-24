import georeferencer.displacement_calc as dc
import numpy as np
import pytest
import time

def generate_synthetic_data(size, dy, dx):
    """
    Generate synthetic data for testing covariance matrix calculation.
    """
    Q = np.random.rand(size, size).astype(np.float32)
    
    Q_shifted = np.roll(Q, (dy, dx), axis=(0, 1))
    return Q_shifted, Q

def test_covariance_matrix_cpp():
    """
    Test the covariance matrix calculation function to ensure consistency between Python and C++ versions.
    """
    size = 48  
    max_displacement = 24 

    swath_P, ref_Q = generate_synthetic_data(size, 23, -23)

    # Test with C++ (through pybind11 interface)
    start_time = time.time()
    cpp_cov_matrix = dc.calculate_covariance_matrix(swath_P, ref_Q, 24)
    cpp_duration = time.time() - start_time
    print(f"C++ covariance matrix calculation time: {cpp_duration:.4f} seconds")
    apex = np.unravel_index(np.argmax(cpp_cov_matrix), cpp_cov_matrix.shape)
    dy, dx = apex
    displacement = (dy - max_displacement, dx - max_displacement)
    assert displacement == (-23, 23), f"Expected (-23, 23) displacement, got {displacement}"

    swath_P, ref_Q = generate_synthetic_data(size, 23, 23)
    cpp_cov_matrix = dc.calculate_covariance_matrix(swath_P, ref_Q, 24)
    apex = np.unravel_index(np.argmax(cpp_cov_matrix), cpp_cov_matrix.shape)
    dy, dx = apex
    displacement = (dy - max_displacement, dx - max_displacement)
    assert displacement == (-23, -23), f"Expected (-23, -23) displacement, got {displacement}"

    swath_P, ref_Q = generate_synthetic_data(size, 0, 0)
    cpp_cov_matrix = dc.calculate_covariance_matrix(swath_P, ref_Q, 24)
    apex = np.unravel_index(np.argmax(cpp_cov_matrix), cpp_cov_matrix.shape)
    dy, dx = apex
    displacement = (dy - max_displacement, dx - max_displacement)
    assert displacement == (0, 0), f"Expected (0, 0) displacement, got {displacement}"

    swath_P, ref_Q = generate_synthetic_data(size, -23, -23)
    cpp_cov_matrix = dc.calculate_covariance_matrix(swath_P, ref_Q, 24)
    apex = np.unravel_index(np.argmax(cpp_cov_matrix), cpp_cov_matrix.shape)
    dy, dx = apex
    displacement = (dy - max_displacement, dx - max_displacement)
    assert displacement == (23, 23), f"Expected (-23, -23) displacement, got {displacement}"

    swath_P, ref_Q = generate_synthetic_data(size, -23, 23)
    cpp_cov_matrix = dc.calculate_covariance_matrix(swath_P, ref_Q, 24)
    apex = np.unravel_index(np.argmax(cpp_cov_matrix), cpp_cov_matrix.shape)
    dy, dx = apex
    displacement = (dy - max_displacement, dx - max_displacement)
    assert displacement == (23, -23), f"Expected (23, -23) displacement, got {displacement}"

def test_covariance_displacement():
    """
    Test the covariance matrix calculation function to ensure that a correct error is returned if no displacement is found
    """
    size = 500
    ref_Q = np.random.rand(size, size).astype(np.float32)
    swath_P = np.roll(ref_Q, (23, -23), axis=(0, 1))

    points = [(250, 250), (200, 250), (400, 400), (100, 100), (100, 250), (50, 70), (60, 95)]

    displacement = dc.calculate_covariance_displacement(points, swath_P, ref_Q, 48, 24)

    assert displacement == (-23, 23), f"Expected (-23, 23) displacement, got {displacement}"

def test_covariance_displacement_no_displacement():
    """
    Test the covariance matrix calculation function to ensure that a correct error is returned if no displacement is found
    """
    size = 500
    ref_Q = np.random.rand(size, size).astype(np.float32)
    swath_P = np.roll(ref_Q, (0, 0), axis=(0, 1))

    points = [(250, 250), (200, 250), (400, 400), (100, 100), (100, 250), (50, 70), (60, 95)]

    displacement = dc.calculate_covariance_displacement(points, swath_P, ref_Q, 48, 24)

    assert displacement == (0, 0), f"Expected (0, 0) displacement, got {displacement}"

def test_covariance_displacement_with_too_large_displacement():
    """
    Test the covariance matrix calculation function to ensure that a correct error is returned if no displacement is found
    """
    size = 500
    ref_Q = np.random.rand(size, size).astype(np.float32)
    swath_P = np.roll(ref_Q, (25, 25), axis=(0, 1))

    points = [(250, 250), (200, 250), (400, 400), (100, 100), (100, 250), (50, 70), (60, 95)]

    displacement = dc.calculate_covariance_displacement(points, swath_P, ref_Q, 48, 24)

    assert displacement == (-100, -100), f"Expected (-100, -100) displacement, got {displacement}"