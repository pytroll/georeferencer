"""Georeferencing and displacement calculation for satellite imagery.

This module provides functions for processing satellite image data,
georeferencing using ground control points (GCPs), and calculating
displacement between a swath image and a reference image.
"""

import logging

import dask.array as da
import numpy as np
import rioxarray
from pyorbital.geoloc_avhrr import estimate_time_and_attitude_deviations
from pyresample import gradient
from pyresample.geometry import AreaDefinition, SwathDefinition
from rasterio.transform import xy
from rasterio.windows import from_bounds
from scipy.spatial import cKDTree

import georeferencer.displacement_calc as dc
import georeferencer.gcp_generation as gcp_gen

logger = logging.getLogger(__name__)
INVALID_DISPLACEMENT = (-100, -100)


def open_subset_tif(filepath, min_lat, max_lat, min_lon, max_lon, buffer=5):
    """Open a subset of a GeoTIFF file based on latitude and longitude bounds.

    This function extracts a subset of a GeoTIFF file by defining a bounding box
    around specified latitude and longitude coordinates, with an optional buffer.

    Args:
        filepath (str): Path to the GeoTIFF file.
        min_lat (float): Minimum latitude for the subset.
        max_lat (float): Maximum latitude for the subset.
        min_lon (float): Minimum longitude for the subset.
        max_lon (float): Maximum longitude for the subset.
        buffer (int, optional): Buffer (in degrees) to extend the bounding box. Default is 5.

    Returns:
        xarray.DataArray: A Dask-backed xarray DataArray containing the subset of the image.
    """
    dataset = rioxarray.open_rasterio(filepath, engine="rasterio")

    dataset_bounds = dataset.rio.bounds()
    dataset_min_lon, dataset_min_lat, dataset_max_lon, dataset_max_lat = dataset_bounds

    min_lat = max(min_lat - buffer, dataset_min_lat)
    max_lat = min(max_lat + buffer, dataset_max_lat)
    min_lon = max(min_lon - buffer, dataset_min_lon)
    max_lon = min(max_lon + buffer, dataset_max_lon)

    subset_window = from_bounds(
        left=min_lon, bottom=min_lat, right=max_lon, top=max_lat, transform=dataset.rio.transform()
    )

    col_off = int(subset_window.col_off)
    row_off = int(subset_window.row_off)
    width = int(subset_window.width)
    height = int(subset_window.height)

    subset = dataset.isel(x=slice(col_off, col_off + width), y=slice(row_off, row_off + height))

    return subset


def gcp_to_lonlat(gcp_x, gcp_y, geo_transform):
    """Convert Ground Control Point (GCP) pixel coordinates to geographic coordinates (longitude, latitude).

    Args:
        gcp_x (int): X-coordinate (pixel) of the GCP.
        gcp_y (int): Y-coordinate (pixel) of the GCP.
        geo_transform (Affine): Geotransform matrix from rasterio.

    Returns:
        tuple: (longitude, latitude) of the GCP in geographic coordinates.
    """
    gcp_x_original = (gcp_x) * 16
    gcp_y_original = (gcp_y) * 16

    xp, yp = xy(geo_transform, gcp_y_original, gcp_x_original)

    return xp, yp


def translate_gcp_to_swath_coordinates(gcp_array, swath_ds, geo_transform):
    """Convert GCP pixel coordinates into swath image coordinates based on latitude and longitude values.

    Args:
        gcp_array (array-like): Array of GCP (x, y) pixel coordinates.
        swath_ds (xarray.Dataset, optional): Dataset containing swath latitude and longitude arrays.
        geo_transform (Affine, optional): Geotransform matrix for coordinate conversion.

    Returns:
        (gcp coords, gcp lonlats): valid GCPs that fall within the swath image bounds, image coords and lonlats.
    """
    longitudes = swath_ds.attrs["area"].lons.values
    latitudes = swath_ds.attrs["area"].lats.values
    image_shape = longitudes.shape

    lon_lat_pairs = np.column_stack((longitudes.ravel(), latitudes.ravel()))
    valid_mask = ~np.isnan(lon_lat_pairs).any(axis=1)
    valid_lon_lat_pairs = lon_lat_pairs[valid_mask]

    # TODO change to another KDTree
    source_geo_def = cKDTree(valid_lon_lat_pairs)

    min_lon = np.nanmin(longitudes)
    max_lon = np.nanmax(longitudes)
    min_lat = np.nanmin(latitudes)
    max_lat = np.nanmax(latitudes)

    nr_of_valid_gcps = 0
    swath_cords = []
    gcp_lonlats = []
    N = 24
    for gcp in gcp_array:
        gcp_lon, gcp_lat = gcp_to_lonlat(gcp[1], gcp[0], geo_transform)

        if gcp_lon < min_lon or gcp_lon > max_lon or gcp_lat < min_lat or gcp_lat > max_lat:
            continue

        distance, index = source_geo_def.query((gcp_lon, gcp_lat))
        swath_index = np.unravel_index(index, image_shape)
        x, y = swath_index
        if N // 2 <= x < image_shape[0] - N // 2 and N // 2 <= y < image_shape[1] - N // 2:
            swath_cords.append(swath_index)
            nr_of_valid_gcps += 1
            gcp_lonlats.append((gcp_lon, gcp_lat))

    return swath_cords, gcp_lonlats


def reproject_reference_into_swath(
    matrix,
    crs,
    swath_longitudes,
    swath_latitudes,
    origin_x,
    origin_y,
    pixel_size_x,
    pixel_size_y,
    min_x,
    min_y,
    max_x,
    max_y,
):
    """Reproject a reference image to match the swath image coordinate system.

    This function aligns a reference image with the swath data by using its
    geospatial bounds, regridding, and resampling it to match the swath.

    Args:
        matrix (np.array): Reference image data as a NumPy array.
        crs (str): Coordinate reference system (CRS) of the reference image.
        swath_longitudes (xarray.DataArray): Swath image longitudes.
        swath_latitudes (xarray.DataArray): Swath image latitudes.
        origin_x (float): X-origin of the reference image.
        origin_y (float): Y-origin of the reference image.
        pixel_size_x (float): Pixel size in the X direction.
        pixel_size_y (float): Pixel size in the Y direction.
        min_x (float): Minimum X coordinate of the reference image.
        min_y (float): Minimum Y coordinate of the reference image.
        max_x (float): Maximum X coordinate of the reference image.
        max_y (float): Maximum Y coordinate of the reference image.

    Returns:
        xarray.DataArray: The reprojected reference image aligned with the swath.
    """
    swath_lons_values = swath_longitudes.values
    swath_lats_values = swath_latitudes.values
    lon_min_swath = np.nanmin(swath_lons_values)
    lon_max_swath = np.nanmax(swath_lons_values)
    lat_min_swath = np.nanmin(swath_lats_values)
    lat_max_swath = np.nanmax(swath_lats_values)

    lon_min_pixel = int(np.floor((lon_min_swath - origin_x) / pixel_size_x))
    lon_max_pixel = int(np.ceil((lon_max_swath - origin_x) / pixel_size_x))
    lat_min_pixel = int(np.floor((origin_y - lat_max_swath) / -pixel_size_y))
    lat_max_pixel = int(np.ceil((origin_y - lat_min_swath) / -pixel_size_y))

    lon_min_source = max(int(min_x), lon_min_swath)
    lon_max_source = min(int(max_x), lon_max_swath)
    lat_min_source = max(int(min_y), lat_min_swath)
    lat_max_source = min(int(max_y), lat_max_swath)

    lon_min_pixel = max(0, lon_min_pixel)
    lon_max_pixel = min(matrix.shape[1], lon_max_pixel)
    lat_min_pixel = max(0, lat_min_pixel)
    lat_max_pixel = min(matrix.shape[0], lat_max_pixel)

    sub_matrix = matrix[lat_min_pixel:lat_max_pixel, lon_min_pixel:lon_max_pixel]
    height, width = sub_matrix.shape
    if height % 2:
        sub_matrix = sub_matrix[:-1, :]
    if width % 2:
        sub_matrix = sub_matrix[:, :-1]

    sub_matrix = np.ascontiguousarray(sub_matrix)
    sub_matrix = gcp_gen.downsample_2x2(sub_matrix)
    sub_matrix = np.asarray(sub_matrix, dtype=np.float32)

    sub_matrix = da.from_array(sub_matrix, chunks=(512, 512))

    source_area = AreaDefinition(
        "subgrid_area",
        "reference image AD around swath lat,lons",
        "tif_projection",
        crs,
        sub_matrix.shape[1],
        sub_matrix.shape[0],
        [lon_min_source, lat_min_source, lon_max_source, lat_max_source],
    )
    full_swath_def = SwathDefinition(lons=swath_longitudes, lats=swath_latitudes)

    res = gradient.create_gradient_search_resampler(source_area, full_swath_def)
    resampled_data = res.resample(sub_matrix)
    return resampled_data


def _build_swath_image(calibrated_ds, sun_zen):
    """Builds a composite swath image using day/night channels corrected for sun zenith."""
    day_swath = calibrated_ds.sel(channel_name="2").channels / np.cos(np.deg2rad(sun_zen))
    night_swath = calibrated_ds.sel(channel_name="4").channels

    max_val = night_swath.max()
    night_swath = max_val - night_swath
    swath = da.where(sun_zen >= 87, night_swath, day_swath)
    return da.nan_to_num(swath).astype(np.float32)


def _load_reference_image(reference_image_path, calibrated_ds):
    """Loads a subset of the reference image based on swath bounds."""
    lons = calibrated_ds["longitude"]
    lats = calibrated_ds["latitude"]
    tif = open_subset_tif(
        reference_image_path,
        np.min(lats),
        np.max(lats),
        np.min(lons),
        np.max(lons),
    )

    ref_image = tif.sel(band=1).values.astype(np.float32)
    height, width = ref_image.shape
    if height % 2:
        ref_image = ref_image[:-1, :]
    if width % 2:
        ref_image = ref_image[:, :-1]

    return ref_image, tif.rio.transform(), tif.rio.bounds(), tif.rio.crs.to_string()


def _reproject_to_swath(ref_image, transform, bounds, calibrated_ds, crs):
    """Reprojects the reference image into the swath coordinate system."""
    return reproject_reference_into_swath(
        ref_image,
        crs,
        calibrated_ds["longitude"],
        calibrated_ds["latitude"],
        transform[2],
        transform[5],
        transform[0],
        transform[4],
        *bounds,
    ).astype(np.float32)


def _generate_gcps(ref_image):
    """Generates ground control point candidates based on variance analysis."""
    STEP = 8
    WINDOW_SIZE = 48
    THINNING_RADIUS = 11
    GROUP_SIZE = 3

    variance_array = gcp_gen.get_variance_array(
        gcp_gen.downsample_2x2(ref_image),
        STEP,
        WINDOW_SIZE,
    )
    return gcp_gen.thin_gcp_candidates(
        variance_array,
        gcp_gen.get_gcp_candidates(variance_array, GROUP_SIZE),
        THINNING_RADIUS,
    )


def _calculate_valid_gcps_from_swath_alignment(swath_coords, gcp_lonlats, swath, ref_swath):
    """Calculates valid GCPs based on displacement analysis between swath and reference."""
    displacement = np.array(
        dc.calculate_covariance_displacement(swath_coords, swath.compute(), ref_swath, 48, 24), dtype=np.float32
    )
    swath_coords = np.array(swath_coords, dtype=np.float32)
    gcp_lonlats = np.array(gcp_lonlats, dtype=np.float32)

    valid_mask = ~(displacement[:, 0] <= INVALID_DISPLACEMENT[0])
    valid_displacements = displacement[valid_mask]
    if len(valid_displacements) == 0:
        raise ValueError("No valid displacements found")
    valid_swath_coords = swath_coords[valid_mask]
    valid_gcp_lonlats = gcp_lonlats[valid_mask]

    valid_gcps = np.column_stack(
        [valid_swath_coords[:, 0] - valid_displacements[:, 0], valid_swath_coords[:, 1] - valid_displacements[:, 1]]
    )
    return valid_gcps, valid_gcp_lonlats


def get_swath_displacement(calibrated_ds, sun_zen, reference_image_path):
    """Calculate the displacement between a swath image and a reference image.

    This function extracts a subset of the reference image, identifies
    ground control points (GCPs), and calculates displacement based on
    covariance analysis.

    Args:
        calibrated_ds (xarray.Dataset): Calibrated swath dataset containing channel data.
        sun_zen (2d Matrix): A 2d matrix with the same shape as the channel data containing the sun zenith angles.
        reference_image_path (str): Path to the reference GeoTIFF image.

    Returns:
        tuple: Displacement values (time difference, roll, pitch, yaw) between the swath and reference image.

    Raises:
        ValueError: If no valid displacement is found.
    """
    swath = _build_swath_image(calibrated_ds, sun_zen)
    ref_image, geo_transform, image_bounds, crs = _load_reference_image(reference_image_path, calibrated_ds)
    target_area = _reproject_to_swath(ref_image, geo_transform, image_bounds, calibrated_ds, crs)
    gcp_points = _generate_gcps(ref_image)
    swath_coords, gcp_lonlats = translate_gcp_to_swath_coordinates(gcp_points, target_area, geo_transform)
    gcps, valid_gcp_lonlats = _calculate_valid_gcps_from_swath_alignment(
        swath_coords, gcp_lonlats, swath, np.nan_to_num(target_area.values, nan=0.0)
    )
    _translate_gcp_lines_to_scanline_offsets(calibrated_ds, gcps)
    logger.debug(f"Found {len(gcps)} valid gcps")
    return estimate_time_and_attitude_deviations(
        gcps,
        valid_gcp_lonlats[:, 0],
        valid_gcp_lonlats[:, 1],
        calibrated_ds["times"][0].values,
        calibrated_ds.attrs["tle"],
        calibrated_ds.attrs["max_scan_angle"],
    )


def _translate_gcp_lines_to_scanline_offsets(calibrated_ds, gcps):
    """Translate gcp line to scanline offsets (zero-based).

    Useful with passes that miss scanlines.
    """
    ints, decs = np.divmod(gcps[:, 0], 1)
    gcps[:, 0] = calibrated_ds.scan_line_index.values[ints.astype(int)] - calibrated_ds.scan_line_index.values[0] + decs


def get_swath_displacement_with_filename(swath_file, tle_dir, tle_file, reference_image_path):
    """Compute swath displacement using a satellite swath file and a reference image.

    This function reads a swath file, retrieves the calibrated dataset,
    and calculates displacement relative to the reference image.

    Args:
        swath_file (str): Path to the swath data file.
        tle_dir (str): Directory containing Two-Line Elements (TLE) files.
        tle_file (str): Name of the specific TLE file to use.
        reference_image_path (str): Path to the reference GeoTIFF image.

    Returns:
        tuple: Displacement values (dx, dy) between the swath and reference image.
    """
    from pygac import get_reader_class

    reader_cls = get_reader_class(swath_file)
    reader = reader_cls(tle_dir=tle_dir, tle_name=tle_file)
    reader.read(swath_file)
    calibrated_ds = reader.get_calibrated_dataset()
    _, _, _, sun_zen, _ = reader.get_angles()

    return get_swath_displacement(calibrated_ds, sun_zen, reference_image_path)
