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


def translate_gcp_to_swath_coordinates(gcp_array, swath_ds=None, geo_transform=None):
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
        gcp_lon, gcp_lat = gcp_to_lonlat(gcp[1], gcp[0], geo_transform.rio.transform())

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
    lon_min_swath = np.nanmin(swath_longitudes.values)
    lon_max_swath = np.nanmax(swath_longitudes.values)
    lat_min_swath = np.nanmin(swath_latitudes.values)
    lat_max_swath = np.nanmax(swath_latitudes.values)

    lon_min_pixel = int((lon_min_swath - origin_x) / pixel_size_x)
    lon_max_pixel = int((lon_max_swath - origin_x) / pixel_size_x)
    lat_min_pixel = int((origin_y - lat_max_swath) / -pixel_size_y)
    lat_max_pixel = int((origin_y - lat_min_swath) / -pixel_size_y)

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

    if height % 2 != 0:
        sub_matrix = sub_matrix[:-1, :]
    if width % 2 != 0:
        sub_matrix = sub_matrix[:, :-1]

    sub_matrix = np.array(gcp_gen.downsample_2x2(sub_matrix), dtype=np.float32)
    sub_matrix = da.from_array(sub_matrix, chunks=(1000, 1000))

    area_id = "subgrid_area"
    description = "reference image AD around swath lat,lons"
    proj_id = "tif_projection"
    proj_dict = crs

    source_area = AreaDefinition(
        area_id,
        description,
        proj_id,
        proj_dict,
        sub_matrix.shape[1],
        sub_matrix.shape[0],
        [lon_min_source, lat_min_source, lon_max_source, lat_max_source],
    )
    full_swath_def = SwathDefinition(lons=swath_longitudes, lats=swath_latitudes)

    res = gradient.create_gradient_search_resampler(source_area, full_swath_def)
    resampled_data = res.resample(sub_matrix)
    return resampled_data


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
    swath = calibrated_ds.sel(channel_name="2").channels.data
    night_swath = calibrated_ds.sel(channel_name="4").channels.data
    max_val = np.nanmax(night_swath)
    night_swath = max_val - night_swath
    swath = np.where(sun_zen >= 90, night_swath, swath)

    swath_longitudes = calibrated_ds["longitude"]
    swath_latitudes = calibrated_ds["latitude"]
    tif = open_subset_tif(
        reference_image_path,
        np.min(swath_latitudes),
        np.max(swath_latitudes),
        np.min(swath_longitudes),
        np.max(swath_longitudes),
    )

    band_data = np.array(tif.sel(band=1).values)
    band_data_f32 = band_data.astype(np.float32)
    height, width = band_data_f32.shape
    if height % 2 != 0:
        band_data_f32 = band_data_f32[:-1, :]
    if width % 2 != 0:
        band_data_f32 = band_data_f32[:, :-1]
    ref_image = gcp_gen.downsample_2x2(band_data_f32)

    variance_array = gcp_gen.get_variance_array(ref_image, 8, 48)
    gcp_candidates = gcp_gen.get_gcp_candidates(variance_array, 3)
    gcp_points = gcp_gen.thin_gcp_candidates(variance_array, gcp_candidates, 11)

    gt = tif.rio.transform()
    origin_x = gt[2]
    pixel_size_x = gt[0]
    origin_y = gt[5]
    pixel_size_y = gt[4]

    min_x, min_y, max_x, max_y = tif.rio.bounds()

    target_area = reproject_reference_into_swath(
        band_data_f32,
        tif.rio.crs.to_string(),
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
    )
    ref_swath = target_area.values
    swath_coords, gcp_lonlats = translate_gcp_to_swath_coordinates(gcp_points, target_area, tif)

    swath = np.nan_to_num(swath, nan=0.0)
    swath = np.asfortranarray(swath, dtype=np.float32)

    ref_swath = np.nan_to_num(ref_swath, nan=0.0)
    ref_swath = np.asfortranarray(ref_swath, dtype=np.float32)

    displacement = dc.calculate_covariance_displacement(swath_coords, swath, ref_swath, 48, 24)
    ref_lons = []
    ref_lats = []
    gcps = []
    for lonlat, coords, disp in zip(gcp_lonlats, swath_coords, displacement, strict=True):
        if disp == INVALID_DISPLACEMENT:
            continue
        ref_lons.append(lonlat[0])
        ref_lats.append(lonlat[1])
        # gcps are line, col
        gcps.append((coords[0] - disp[0], coords[1] - disp[1]))

    gcps = np.array(gcps)
    logger.debug(f"Found {len(gcps)} valid daytime gcps")
    return estimate_time_and_attitude_deviations(
        gcps, ref_lons, ref_lats, calibrated_ds["times"][0].values, calibrated_ds.attrs["tle"], 55.37
    )


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
