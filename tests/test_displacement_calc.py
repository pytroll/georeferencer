"""Tests for displacement_calc library."""

import math
from pathlib import Path

import georeferencer.displacement_calc as dc
import numpy as np
from scipy.ndimage import gaussian_filter


def generate_synthetic_data(size, dy, dx):
    """Generate synthetic data for testing covariance matrix calculation."""
    rng = np.random.default_rng()
    Q = rng.random((size, size), dtype=np.float32)
    Q_shifted = np.roll(Q, (dy, dx), axis=(0, 1))
    return Q_shifted, Q


def test_covariance_matrix_cpp():
    """Test the covariance matrix calculation function to ensure that a correct displacement is returned."""
    size = 48
    max_displacement = 24

    swath_P, ref_Q = generate_synthetic_data(size, 23, -23)

    cpp_cov_matrix = dc.calculate_covariance_matrix(swath_P, ref_Q, 24)
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
    """Test the covariance matrix calculation function to ensure that correct value is returned."""
    size = 500
    rng = np.random.default_rng()
    ref_Q = rng.random((size, size), dtype=np.float32)
    swath_P = np.roll(ref_Q, (23, -23), axis=(0, 1))

    points = [(250, 250), (200, 250), (400, 400), (100, 100), (100, 250), (50, 70), (60, 95)]

    displacement = dc.calculate_covariance_displacement(points, swath_P, ref_Q, 48, 24)

    tol = 1e-1
    for i, (dy, dx) in enumerate(displacement):
        assert math.isclose(dy, -23.0, abs_tol=tol), f"displacement[{i}].dy = {dy:.3e} exceeds ±{tol}"
        assert math.isclose(dx, 23.0, abs_tol=tol), f"displacement[{i}].dx = {dx:.3e} exceeds ±{tol}"


def test_covariance_displacement_no_displacement():
    """Test the covariance matrix calculation function to ensure that correct value is returned."""
    size = 500
    rng = np.random.default_rng()
    ref_Q = rng.random((size, size), dtype=np.float32)
    swath_P = np.roll(ref_Q, (0, 0), axis=(0, 1))

    points = [(250, 250), (200, 250), (400, 400), (100, 100), (100, 250), (50, 70), (60, 95)]

    displacement = dc.calculate_covariance_displacement(points, swath_P, ref_Q, 48, 24)

    tol = 1e-4
    for i, (dy, dx) in enumerate(displacement):
        assert math.isclose(dy, 0.0, abs_tol=tol), f"displacement[{i}].dy = {dy:.3e} exceeds ±{tol}"
        assert math.isclose(dx, 0.0, abs_tol=tol), f"displacement[{i}].dx = {dx:.3e} exceeds ±{tol}"


def test_covariance_displacement_with_too_large_displacement():
    """Test the covariance matrix calculation function to ensure that a correct error is returned."""
    size = 500
    rng = np.random.default_rng()
    ref_Q = rng.random((size, size), dtype=np.float32)
    swath_P = np.roll(ref_Q, (25, 25), axis=(0, 1))

    points = [(250, 250), (200, 250), (400, 400), (100, 100), (100, 250), (50, 70), (60, 95)]

    displacement = dc.calculate_covariance_displacement(points, swath_P, ref_Q, 48, 24)

    for i, (dy, dx) in enumerate(displacement):
        assert math.isclose(dy, np.float32(-100.0), abs_tol=0), f"displacement[{i}].dy = {dy:.3e}"
        assert math.isclose(dx, np.float32(-100.0), abs_tol=0), f"displacement[{i}].dx = {dx:.3e}"


def downsample_mean(img: np.ndarray, factor: int):
    """Downsample image by averaging over non-overlapping blocks of shape (factor x factor)."""
    H, W = img.shape
    reshaped = img[: H // factor * factor, : W // factor * factor]
    reshaped = reshaped.reshape(H // factor, factor, W // factor, factor)
    down = reshaped.mean(axis=(1, 3))
    return down


def downsample_mean_offset(img: np.ndarray, factor: int, dy: float, dx: float):
    """Downsample the image with an offset applied before averaging."""
    oy = int(round(dy * factor))
    ox = int(round(dx * factor))
    H, W = img.shape
    H_crop = (H - oy) // factor * factor
    W_crop = (W - ox) // factor * factor
    sub = img[oy : oy + H_crop, ox : ox + W_crop]
    return downsample_mean(sub, factor)


def get_or_create_test_image(path: Path, shape=(10000, 10000), min_val=0.0, max_val=255.0, seed=42, add_pattern=True):
    """Load an image from disk if it exists, or create and save a random structured image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return np.load(path)

    rng = np.random.default_rng(seed)
    img = rng.uniform(min_val, max_val, size=shape).astype(np.float32)

    if add_pattern:
        y = np.linspace(0, 10 * np.pi, shape[0], dtype=np.float32)
        x = np.linspace(0, 10 * np.pi, shape[1], dtype=np.float32)
        X, Y = np.meshgrid(x, y)
        pattern = 50 * np.sin(X) * np.cos(Y)
        img += pattern.astype(np.float32)

    img = gaussian_filter(img, sigma=2)

    img = (img - img.min()) / (img.max() - img.min()) * (max_val - min_val) + min_val

    np.save(path, img)
    return img


def test_sub_pixel_calculations_A():
    """Test sub pixel calculations with two low value decimals."""
    img_file = Path(__file__).parent / "test_img.npy"
    img = get_or_create_test_image(img_file)
    F = 10

    shift_A = (0.0, 0.0)
    shift_B = (5.2, 3.1)

    A = downsample_mean_offset(img, F, *shift_A)
    B = downsample_mean_offset(img, F, *shift_B)

    points = [(250, 250), (200, 250), (400, 400), (100, 100), (100, 250), (50, 70), (60, 95)]

    displacement = dc.calculate_covariance_displacement(points, B, A, 48, 24)
    tol = 0.3
    for i, (dy, dx) in enumerate(displacement):
        assert math.isclose(dy, shift_B[0], abs_tol=tol), f"displacement[{i}].dy = {dy:.3e} exceeds ±{tol}"
        assert math.isclose(dx, shift_B[1], abs_tol=tol), f"displacement[{i}].dx = {dx:.3e} exceeds ±{tol}"


def test_sub_pixel_calculations_B():
    """Test sub pixel calculations with one low and one high value decimal."""
    img_file = Path(__file__).parent / "test_img.npy"
    img = get_or_create_test_image(img_file)
    F = 10

    shift_A = (0.0, 0.0)
    shift_B = (3.2, 1.8)

    A = downsample_mean_offset(img, F, *shift_A)
    B = downsample_mean_offset(img, F, *shift_B)

    points = [(250, 250), (200, 250), (400, 400), (100, 100), (100, 250), (50, 70), (60, 95)]

    displacement = dc.calculate_covariance_displacement(points, B, A, 48, 24)
    tol = 0.3
    for i, (dy, dx) in enumerate(displacement):
        assert math.isclose(dy, shift_B[0], abs_tol=tol), f"displacement[{i}].dy = {dy:.3e} exceeds ±{tol}"
        assert math.isclose(dx, shift_B[1], abs_tol=tol), f"displacement[{i}].dx = {dx:.3e} exceeds ±{tol}"


def test_sub_pixel_calculations_C():
    """Test sub pixel calculations with one high and one low value decimal."""
    img_file = Path(__file__).parent / "test_img.npy"
    img = get_or_create_test_image(img_file)
    F = 10

    shift_A = (0.0, 0.0)
    shift_B = (7.9, 2.2)

    A = downsample_mean_offset(img, F, *shift_A)
    B = downsample_mean_offset(img, F, *shift_B)

    points = [(250, 250), (200, 250), (400, 400), (100, 100), (100, 250), (50, 70), (60, 95)]

    displacement = dc.calculate_covariance_displacement(points, B, A, 48, 24)
    tol = 0.3
    for i, (dy, dx) in enumerate(displacement):
        assert math.isclose(dy, shift_B[0], abs_tol=tol), f"displacement[{i}].dy = {dy:.3e} exceeds ±{tol}"
        assert math.isclose(dx, shift_B[1], abs_tol=tol), f"displacement[{i}].dx = {dx:.3e} exceeds ±{tol}"


def test_sub_pixel_calculations_D():
    """Test sub pixel calculations with two high value decimal."""
    img_file = Path(__file__).parent / "test_img.npy"
    img = get_or_create_test_image(img_file)
    F = 10

    shift_A = (0.0, 0.0)
    shift_B = (5.8, 6.9)

    A = downsample_mean_offset(img, F, *shift_A)
    B = downsample_mean_offset(img, F, *shift_B)

    points = [(250, 250), (200, 250), (400, 400), (100, 100), (100, 250), (50, 70), (60, 95)]

    displacement = dc.calculate_covariance_displacement(points, B, A, 48, 24)
    tol = 0.3
    for i, (dy, dx) in enumerate(displacement):
        assert math.isclose(dy, shift_B[0], abs_tol=tol), f"displacement[{i}].dy = {dy:.3e} exceeds ±{tol}"
        assert math.isclose(dx, shift_B[1], abs_tol=tol), f"displacement[{i}].dx = {dx:.3e} exceeds ±{tol}"


def test_sub_pixel_calculations_E():
    """Test sub pixel calculations with two mid range value decimal."""
    img_file = Path(__file__).parent / "test_img.npy"
    img = get_or_create_test_image(img_file)
    F = 10

    shift_A = (0.0, 0.0)
    shift_B = (5.7, 6.3)

    A = downsample_mean_offset(img, F, *shift_A)
    B = downsample_mean_offset(img, F, *shift_B)

    points = [(250, 250), (200, 250), (400, 400), (100, 100), (100, 250), (50, 70), (60, 95)]

    displacement = dc.calculate_covariance_displacement(points, B, A, 48, 24)
    tol = 0.3
    for i, (dy, dx) in enumerate(displacement):
        assert math.isclose(dy, shift_B[0], abs_tol=tol), f"displacement[{i}].dy = {dy:.3e} exceeds ±{tol}"
        assert math.isclose(dx, shift_B[1], abs_tol=tol), f"displacement[{i}].dx = {dx:.3e} exceeds ±{tol}"


def test_sub_pixel_calculations_F():
    """Test sub pixel calculations with two mid range value decimal."""
    img_file = Path(__file__).parent / "test_img.npy"
    img = get_or_create_test_image(img_file)
    F = 10

    shift_A = (0.0, 0.0)
    shift_B = (5.4, 6.7)

    A = downsample_mean_offset(img, F, *shift_A)
    B = downsample_mean_offset(img, F, *shift_B)

    points = [(250, 250), (200, 250), (400, 400), (100, 100), (100, 250), (50, 70), (60, 95)]

    displacement = dc.calculate_covariance_displacement(points, B, A, 48, 24)
    tol = 0.3
    for i, (dy, dx) in enumerate(displacement):
        assert math.isclose(dy, shift_B[0], abs_tol=tol), f"displacement[{i}].dy = {dy:.3e} exceeds ±{tol}"
        assert math.isclose(dx, shift_B[1], abs_tol=tol), f"displacement[{i}].dx = {dx:.3e} exceeds ±{tol}"
