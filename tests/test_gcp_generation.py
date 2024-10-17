#import georeferencer.gcp_generation as gcp_gen
import georeferencer.gcp_generation as gcp_gen
import numpy as np
import pytest

def test_downsample_empty_matrix():
    with pytest.raises(ValueError):
        matrix = np.array([])
        gcp_gen.downsample_2x2(matrix)

def test_downsample_single_block():
    matrix = np.array([
        [1, 2],
        [3, 4]
    ])
    expected_output = np.array([[2.5]])
    result = gcp_gen.downsample_2x2(matrix)
    assert result == expected_output, f"Expected {expected_output}, got {result}"

def test_downsample_even_matrix():
    matrix = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])
    expected_output = np.array([
        [3.5, 5.5],
        [11.5, 13.5]
    ])
    result = gcp_gen.downsample_2x2(matrix)
    assert np.array_equal(result, expected_output), f"Expected {expected_output}, got {result}"

def test_variance_output_shape():
    pixel_matrix = np.random.rand(100, 100)
    step = 8
    box_size = 48
    expected_shape = ((100 - box_size) // step + 1, (100 - box_size) // step + 1)
    
    result = gcp_gen.get_variance_array(pixel_matrix, step=step, box_size=box_size)
    assert result.shape == expected_shape

def test_variance_constant_matrix():
    pixel_matrix = np.ones((64, 64))
    result = gcp_gen.get_variance_array(pixel_matrix, step=8, box_size=48)
    
    assert np.all(result == 0)

def test_variance_small_matrix():
    pixel_matrix = np.array([[1, 2, 3, 4, 5],
                             [2, 3, 4, 5, 6],
                             [3, 4, 5, 6, 7],
                             [4, 5, 6, 7, 8],
                             [5, 6, 7, 8, 9]], dtype=np.float32)
    
    result = gcp_gen.get_variance_array(pixel_matrix, step=1, box_size=3)
    print(result)
    
    expected_variance = 1.33
    assert pytest.approx(result[0, 0], 0.1) == expected_variance

def test_non_uniform_matrix():
    pixel_matrix = np.array([[0, 0, 0, 0],
                             [0, 100, 100, 0],
                             [0, 100, 100, 0],
                             [0, 0, 0, 0]], dtype=np.float32)
    
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
    with pytest.raises(ValueError):
        gcp_gen.get_variance_array(np.zeros(shape=(0, 0)))

def test_calculate_variance_all_ones():
    reflectance_box = np.ones((48, 48), dtype=np.float32)
    variance = gcp_gen.get_variance_array(reflectance_box, 1, 48)
    assert variance == 0.0, f"Expected 0 variance, got {variance}"

def test_calculate_variance_random():
    reflectance_box = np.random.rand(48,48)
    variance = gcp_gen.get_variance_array(reflectance_box, 1, 48)
    assert variance >= 0.0, f"Expected non negative variance, got {variance}"

def test_calculate_variance_zeroes():
    reflectance_box = np.zeros((48, 48), dtype=np.float32)
    variance = gcp_gen.get_variance_array(reflectance_box, 1, 48)
    assert variance == 0.0, f"Expected 0 variance, got {variance}"

def test_calculate_variance():
    sample_box = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8]
    ], dtype=np.float32)
    expected_variance = 5.25
    result = gcp_gen.get_variance_array(sample_box, step=1, box_size=8)
    assert np.isclose(result[0][0], expected_variance, rtol=1e-5), f"Expected {expected_variance}, got {result}"  

def test_gcp_candidates_basic():
    variance_array = np.array([[1, 2, 3],
                               [4, 5, 6],
                               [7, 8, 9]])
    
    result = gcp_gen.get_gcp_candidates(variance_array, group_size=3)
    
    assert result == [(2, 2)]

def test_gcp_candidates_all_zeros():
    # Test case with all zeros
    variance_array = np.zeros((6, 6))
    
    result = gcp_gen.get_gcp_candidates(variance_array, group_size=3)
    
    # Even though all values are zero, it should return the top-left corner of each group
    expected_result = [(0, 0), (0, 3), (3, 0), (3, 3)]
    assert result == expected_result

def test_gcp_candidates_multiple_maxima():
    # Test case where there are multiple maxima in the groups
    variance_array = np.array([[1, 2, 3],
                               [4, 9, 6],
                               [7, 8, 9]])
    
    result = gcp_gen.get_gcp_candidates(variance_array, group_size=3)
    
    assert result == [(1, 1)]

def test_gcp_candidates_smaller_than_group():
    variance_array = np.array([[1, 2],
                               [3, 4]])
    
    result = gcp_gen.get_gcp_candidates(variance_array, group_size=3)
    
    assert result == []

def test_gcp_candidates_multiple_groups():
    variance_array = np.array([[1, 2, 3, 1, 2, 3],
                               [4, 5, 6, 4, 5, 6],
                               [7, 8, 9, 7, 8, 9],
                               [1, 2, 3, 1, 2, 3],
                               [4, 5, 6, 4, 5, 6],
                               [7, 8, 9, 7, 8, 9]])
    
    result = gcp_gen.get_gcp_candidates(variance_array, group_size=3)
    
    expected_result = [(2, 2), (2, 5), (5, 2), (5, 5)]
    assert result == expected_result

def test_thin_gcp_candidates_basic():
    variance_array = np.array([[1, 1, 1, 1, 1],
                               [1, 2, 2, 2, 1],
                               [1, 2, 9, 2, 1],
                               [1, 2, 2, 2, 1],
                               [1, 1, 1, 1, 1]])
    
    gcp_candidates = [(2, 2)]
    
    result = gcp_gen.thin_gcp_candidates(variance_array, gcp_candidates, group_size=3)
    assert result == [(2, 2)]

def test_thin_gcp_candidates_reject_candidate():
    variance_array = np.array([[1, 1, 1, 1, 1],
                               [1, 2, 2, 2, 1],
                               [1, 2, 9, 2, 1],
                               [1, 2, 2, 2, 1],
                               [1, 1, 1, 1, 1]])
    
    gcp_candidates = [(2, 2), (1, 1)]  # Candidate (1,1) should be rejected
    
    result = gcp_gen.thin_gcp_candidates(variance_array, gcp_candidates, group_size=3)
    assert result == [(2, 2)]

def test_thin_gcp_candidates_multiple_maxima():
    # Case where multiple maxima exist in a group
    variance_array = np.array([[1, 1, 1, 1, 1],
                               [1, 9, 2, 9, 1],
                               [1, 2, 1, 2, 1],
                               [1, 9, 2, 9, 1],
                               [1, 1, 1, 1, 1]])
    
    gcp_candidates = [(1, 1), (1, 3), (3, 1), (3, 3)]
    
    result = gcp_gen.thin_gcp_candidates(variance_array, gcp_candidates, group_size=3)
    
    
    # Only one maximum per group should be selected
    assert result == [(1, 1), (1, 3), (3, 1), (3, 3)]

def test_thin_gcp_candidates_edge_case():
    variance_array = np.array([[9, 1, 1],
                               [1, 2, 1],
                               [1, 1, 9]])
    
    gcp_candidates = [(0, 0), (2, 2)]
    
    result = gcp_gen.thin_gcp_candidates(variance_array, gcp_candidates, group_size=3)
    
    assert result == [(0, 0), (2, 2)]

def test_thin_gcp_candidates_reject_most_but_recover_one_with_condition():
    variance_array = np.array([[1, 10, 1, 1, 1],
                               [1, 2, 3, 4, 5],
                               [6, 1, 8, 9, 10],
                               [11, 1, 1, 1, 15],
                               [16, 17, 18, 19, 20]])
    
    gcp_candidates = [(0, 1), (0, 2),(1, 1), (1, 2), (2, 2), (1, 4), (1, 1), (1, 2), (2, 3), (2, 4)]

    result = gcp_gen.thin_gcp_candidates(variance_array, gcp_candidates, group_size=3)

    #(0, 1): should be retained because it's local maxima
    #(2, 4) : should be kept thanks to recovery condition
    #(2, 2): should be kept thanks to recovery condition
    assert result == [(0, 1), (2, 4), (2, 2)]

def test_thin_gcp_candidates_large_group_size():
    variance_array = np.array([[9, 1, 1, 1, 1, 1],
                               [1, 1, 1, 9, 1, 1],
                               [1, 1, 1, 1, 1, 1],
                               [2, 9, 1, 9, 1, 1],
                               [1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1]])
    
    gcp_candidates = [(0, 0), (1, 3), (3, 1), (3, 3)]
    result = gcp_gen.thin_gcp_candidates(variance_array, gcp_candidates, group_size=5)

    #(0,0) should be selected immediately and (3,1) should be recovered
    assert result == [(0, 0), (3,1)]
