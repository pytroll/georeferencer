"""Tests for gcp_generation."""

import georeferencer.gcp_generation as gcp_gen
import numpy as np
import pytest


def test_downsample_empty_matrix():
    """Test that an empty matrix raises an error."""
    matrix = np.array([])
    with pytest.raises(ValueError, match="not enough values to unpack"):
        gcp_gen.downsample_2x2(matrix)


def test_downsample_single_block():
    """Test downsample 2x2 matrix."""
    matrix = np.array([[1, 2], [3, 4]])
    expected_output = np.array([[2.5]])
    result = gcp_gen.downsample_2x2(matrix)
    assert result == expected_output, f"Expected {expected_output}, got {result}"


def test_downsample_even_matrix():
    """Test downsample."""
    matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    expected_output = np.array([[3.5, 5.5], [11.5, 13.5]])
    result = gcp_gen.downsample_2x2(matrix)
    assert np.array_equal(result, expected_output), f"Expected {expected_output}, got {result}"


def test_variance_output_shape():
    """Make sure output shape is correct."""
    rng = np.random.default_rng()
    pixel_matrix = rng.random((100, 100), dtype=np.float32)
    step = 8
    box_size = 48
    expected_shape = ((100 - box_size) // step + 1, (100 - box_size) // step + 1)

    result = gcp_gen.get_variance_array(pixel_matrix, step=step, box_size=box_size)
    assert result.shape == expected_shape


def test_variance_constant_matrix():
    """Test variance calculations with all ones."""
    pixel_matrix = np.ones((64, 64))
    result = gcp_gen.get_variance_array(pixel_matrix, step=8, box_size=48)

    assert np.all(result == 0)


def test_variance_small_matrix():
    """Test variance calculation with small matrix."""
    pixel_matrix = np.array(
        [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [5, 6, 7, 8, 9]], dtype=np.float32
    )

    result = gcp_gen.get_variance_array(pixel_matrix, step=1, box_size=3)
    expected_variance = 1.33
    assert pytest.approx(result[0, 0], 0.1) == expected_variance


def test_non_uniform_matrix():
    """This test checks if the function correctly calculates variance in different regions of a small 4x4 pixel matrix.

    The matrix contains a high-intensity (100) square in the center, surrounded by zeros.
    Expected variance values are checked at corners, edges, and the center.
    """
    pixel_matrix = np.array([[0, 0, 0, 0], [0, 100, 100, 0], [0, 100, 100, 0], [0, 0, 0, 0]], dtype=np.float32)

    result = gcp_gen.get_variance_array(pixel_matrix, step=1, box_size=2)

    expected_variance_edge = 2500
    expected_variance_corner = 1875
    expected_variance_center = 0
    assert pytest.approx(result[0, 0], 1) == expected_variance_corner
    assert pytest.approx(result[0, 1], 1) == expected_variance_edge
    assert pytest.approx(result[1, 0], 1) == expected_variance_edge
    assert pytest.approx(result[1, 1], 1) == expected_variance_center
    assert pytest.approx(result[1, 2], 1) == expected_variance_edge
    assert pytest.approx(result[0, 2], 1) == expected_variance_corner


def test_get_empty_variance_array():
    """Check that an error is raised with an empty matrix."""
    with pytest.raises(ValueError, match="negative dimensions are not allowed"):
        gcp_gen.get_variance_array(np.zeros(shape=(0, 0)))


def test_calculate_variance_all_ones():
    """Calculate variance from a matrix with all ones."""
    reflectance_box = np.ones((48, 48), dtype=np.float32)
    variance = gcp_gen.get_variance_array(reflectance_box, 1, 48)
    assert variance == 0.0, f"Expected 0 variance, got {variance}"


def test_calculate_variance_random():
    """Make sure variance is not zero."""
    rng = np.random.default_rng()
    reflectance_box = rng.random((48, 48), dtype=np.float32)
    variance = gcp_gen.get_variance_array(reflectance_box, 1, 48)
    assert variance >= 0.0, f"Expected non negative variance, got {variance}"


def test_calculate_variance_zeroes():
    """Make sure variance is zero with all zeroes."""
    reflectance_box = np.zeros((48, 48), dtype=np.float32)
    variance = gcp_gen.get_variance_array(reflectance_box, 1, 48)
    assert variance == 0.0, f"Expected 0 variance, got {variance}"


def test_calculate_variance():
    """Test variance calculation with a small matrix."""
    sample_box = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ],
        dtype=np.float32,
    )
    expected_variance = 5.25
    result = gcp_gen.get_variance_array(sample_box, step=1, box_size=8)
    assert np.isclose(result[0][0], expected_variance, rtol=1e-5), f"Expected {expected_variance}, got {result}"


def test_gcp_candidates_basic():
    """Test ground control point generation with a small matrix."""
    variance_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    result = gcp_gen.get_gcp_candidates(variance_array, group_size=3)

    assert result == [(2, 2)]


def test_gcp_candidates_all_zeros():
    """Even though all values are zero, it should return the top-left corner of each group."""
    variance_array = np.zeros((6, 6))

    result = gcp_gen.get_gcp_candidates(variance_array, group_size=3)

    expected_result = [(0, 0), (0, 3), (3, 0), (3, 3)]
    assert result == expected_result


def test_gcp_candidates_multiple_maxima():
    """Test case where there are multiple maxima in the groups."""
    variance_array = np.array([[1, 2, 3], [4, 9, 6], [7, 8, 9]])

    result = gcp_gen.get_gcp_candidates(variance_array, group_size=3)

    assert result == [(1, 1)]


def test_gcp_candidates_smaller_than_group():
    """Make sure no candidate is returned with an array that is smaller than group size."""
    variance_array = np.array([[1, 2], [3, 4]])

    result = gcp_gen.get_gcp_candidates(variance_array, group_size=3)

    assert result == []


def test_gcp_candidates_multiple_groups():
    """Make sure multiple correct ground control points are discovered."""
    variance_array = np.array(
        [
            [1, 2, 3, 1, 2, 3],
            [4, 5, 6, 4, 5, 6],
            [7, 8, 9, 7, 8, 9],
            [1, 2, 3, 1, 2, 3],
            [4, 5, 6, 4, 5, 6],
            [7, 8, 9, 7, 8, 9],
        ]
    )

    result = gcp_gen.get_gcp_candidates(variance_array, group_size=3)

    expected_result = [(2, 2), (2, 5), (5, 2), (5, 5)]
    assert result == expected_result


def test_thin_gcp_candidates_basic():
    """Make sure no candidates is removed when it's the local maxima."""
    variance_array = np.array([[1, 1, 1, 1, 1], [1, 2, 2, 2, 1], [1, 2, 9, 2, 1], [1, 2, 2, 2, 1], [1, 1, 1, 1, 1]])

    gcp_candidates = [(2, 2)]

    result = gcp_gen.thin_gcp_candidates(variance_array, gcp_candidates, group_size=3)
    assert result == [(2, 2)]


def test_thin_gcp_candidates_reject_candidate():
    """Make sure candidate is removed wehn it is not the local maxima."""
    variance_array = np.array([[1, 1, 1, 1, 1], [1, 2, 2, 2, 1], [1, 2, 9, 2, 1], [1, 2, 2, 2, 1], [1, 1, 1, 1, 1]])

    gcp_candidates = [(2, 2), (1, 1)]

    result = gcp_gen.thin_gcp_candidates(variance_array, gcp_candidates, group_size=3)
    assert result == [(2, 2)]


def test_thin_gcp_candidates_multiple_maxima():
    """Make sure correct candidates are removed with multiple local maxima."""
    variance_array = np.array([[1, 1, 1, 1, 1], [1, 9, 2, 9, 1], [1, 2, 1, 2, 1], [1, 9, 2, 9, 1], [1, 1, 1, 1, 1]])

    gcp_candidates = [(1, 1), (1, 2), (1, 3), (2, 1), (3, 1), (3, 2), (3, 3)]

    result = gcp_gen.thin_gcp_candidates(variance_array, gcp_candidates, group_size=3)

    assert result == [(1, 1), (1, 3), (3, 1), (3, 3)]


def test_thin_gcp_candidates_edge_case():
    """Test case with maxima at edges."""
    variance_array = np.array([[9, 1, 1], [1, 2, 1], [1, 1, 9]])

    gcp_candidates = [(0, 0), (2, 2)]

    result = gcp_gen.thin_gcp_candidates(variance_array, gcp_candidates, group_size=3)

    assert result == [(0, 0), (2, 2)]


def test_thin_gcp_candidates_reject_most_but_recover_one_with_condition():
    """Test removal and recovery.

    (0, 1): should be retained because it's local maxima
    (2, 4): should be kept thanks to recovery condition
    (2, 2): should be kept thanks to recovery condition
    """
    variance_array = np.array(
        [[1, 10, 1, 1, 1], [1, 2, 3, 4, 5], [6, 1, 8, 9, 10], [11, 1, 1, 1, 15], [16, 17, 18, 19, 20]]
    )

    gcp_candidates = [(0, 1), (0, 2), (1, 1), (1, 2), (2, 2), (1, 4), (1, 1), (1, 2), (2, 3), (2, 4)]

    result = gcp_gen.thin_gcp_candidates(variance_array, gcp_candidates, group_size=3)

    assert result == [(0, 1), (2, 4), (2, 2)]


def test_thin_gcp_candidates_large_group_size():
    """Test removal and recovery with large group size.

    (0,0): should be retained because it's local maxima
    (3,1): should be kept thanks to recovery condition
    """
    variance_array = np.array(
        [
            [9, 1, 1, 1, 1, 1],
            [1, 1, 1, 9, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [2, 9, 1, 9, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ]
    )

    gcp_candidates = [(0, 0), (1, 3), (3, 1), (3, 3)]
    result = gcp_gen.thin_gcp_candidates(variance_array, gcp_candidates, group_size=5)

    assert result == [(0, 0), (3, 1)]
