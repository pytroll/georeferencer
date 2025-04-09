"""Generating and handling Ground control points."""

import numpy as np
from scipy import ndimage


def downsample_2x2(matrix):
    """Reduces the resolution of matrix by averaging 2x2 pixels into 1 pixel.

    Parameters:
    matrix (np.array): A pixel matrix

    Returns:
    (np.array): Downsampled pixel matrix
    """
    rows, cols = matrix.shape

    if rows % 2 != 0 or cols % 2 != 0:
        raise ValueError("The pixel matrix dimensions must be divisible by 2.")

    reshaped_matrix = matrix.reshape(rows // 2, 2, cols // 2, 2)
    downsampled_matrix = reshaped_matrix.mean(axis=(1, 3))

    return downsampled_matrix


def get_variance_array_from_file(path="gcp_library.npz"):
    """Load the variance array from a specified NPZ file.

    Parameters:
    path (str): The file path to the NPZ file containing the variance array.
                Default is 'gcp_library.npz'.

    Returns:
    np.ndarray: An array of variance values loaded from the NPZ file.
                The array is expected to be stored under the key 'variance_array'.
    """
    return np.load(path, allow_pickle=True)["variance_array"]


def save_reference_data(gcp_points, variance_array, path="gcp_library.npz"):
    """Save GCP (Ground Control Points) and variance array to an NPZ file.

    Parameters:
    gcp_points (np.ndarray): An array of ground control points to be saved.
    variance_array (np.ndarray): An array of variance values corresponding to the GCPs.
    path (str): The file path where the NPZ file will be saved.
                Default is 'gcp_library.npz'.

    Returns:
    None: This function does not return any value. It saves the data to the specified file.
    """
    np.savez(path, gcp_points=gcp_points, variance_array=variance_array)


def get_gcp_points_from_file(path="gcp_library.npz"):
    """Load GCP (Ground Control Points) from a specified NPZ file.

    Parameters:
    path (str): The file path to the NPZ file containing the GCP points.
                Default is 'gcp_library.npz'.

    Returns:
    np.ndarray: An array of GCP points loaded from the NPZ file.
                The array is expected to contain the GCP points stored
                under the key 'gcp_points'.
    """
    return np.load(path, allow_pickle=True)["gcp_points"]


def get_variance_array(matrix, step=8, box_size=48):
    """Compute the variance of reflectance values in a matrix.

    Parameters:
    matrix (np.array, dtype=np.float32): A matrix containing reflectance values
    step (np.int) : How often the variance is calculated per pixel/line
    box_size (np.int) : How many pixels in box_size x box_size in which variance is calculated from

    Returns:
    variance_array (np.array): An array of variances
    """
    window_size = (box_size, box_size)
    height, width = matrix.shape
    variance_array = np.zeros(((height - box_size) // step + 1, (width - box_size) // step + 1), dtype=np.float32)

    win_mean = ndimage.uniform_filter(matrix, window_size)
    win_sqr_mean = ndimage.uniform_filter(matrix**2, window_size)
    variance = win_sqr_mean - win_mean**2

    for i in range(0, height - box_size + 1, step):
        for j in range(0, width - box_size + 1, step):
            variance_array[i // step, j // step] = variance[i + box_size // 2][j + box_size // 2]
    return variance_array


def get_gcp_candidates(variance_array, group_size=3):
    """Generates a list of potential ground control points.

    Parameters:
    variance_array (np.array): A matrix containing variance values
    group_size (np.int) : How many pixels in group_size x group_size in which candidates are chosen from

    Returns:
    list: A list of indices of potential gcp candidates in the variance_array
    """
    gcp_candidates = []
    height, width = variance_array.shape
    # TODO find np functions to remove nested for loops
    for i in range(0, height - group_size + 1, group_size):
        for j in range(0, width - group_size + 1, group_size):
            group = variance_array[i : i + group_size, j : j + group_size]
            pointIndex = np.unravel_index(np.argmax(group), group.shape)
            gcp_candidates.append((pointIndex[0] + i, pointIndex[1] + j))
    return gcp_candidates


def thin_gcp_candidates(variance_array, gcp_candidates, group_size=11):
    """Removes potential ground control points with low quality.

    Parameters:
    variance_array (np.array): A matrix containing variance values
    gcp_candidates (list): An array containing indices (i,j) for gcp points in variance_array
    group_size (np.int) : How many pixels in group_size x group_size in which candidates are chosen

    Returns:
    thinned_gcp_candidates (list): A list of indices of gcp candidates in the variance_array
    """
    thinned_gcp_candidates = []
    half_group = (group_size - 1) // 2
    height, width = variance_array.shape
    good_gcp_mask = np.zeros(variance_array.shape, dtype=bool)

    for i, j in gcp_candidates:
        box = variance_array[
            max(0, i - half_group) : min(height, i + half_group + 1),
            max(0, j - half_group) : min(width, j + half_group + 1),
        ]
        center_value = variance_array[i, j]

        if center_value == np.max(box) and np.sum(box == center_value) == 1:
            thinned_gcp_candidates.append((i, j))
            good_gcp_mask[i, j] = True

    discarded_gcp_candidates = set(gcp_candidates) - set(thinned_gcp_candidates)

    for i, j in discarded_gcp_candidates:
        box_min_i = max(0, i - half_group)
        box_max_i = min(height, i + half_group + 1)
        box_min_j = max(0, j - half_group)
        box_max_j = min(width, j + half_group + 1)

        neighbor_box = good_gcp_mask[box_min_i:box_max_i, box_min_j:box_max_j]
        has_good_neighbours = np.any(neighbor_box)

        if not has_good_neighbours:
            box = variance_array[box_min_i:box_max_i, box_min_j:box_max_j]
            center_value = variance_array[i, j]
            box_max = np.max(box)
            box_min = np.min(box)
            box_mean = np.mean(box)

            if (center_value**2) > ((box_mean**2) + (0.15 * (box_max**2 - box_min**2))):
                thinned_gcp_candidates.append((i, j))
                good_gcp_mask[i, j] = True

    return thinned_gcp_candidates
