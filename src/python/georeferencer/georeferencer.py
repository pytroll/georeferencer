"""Georeferencing and displacement calculation for satellite imagery.

This module provides functions for processing satellite image data,
georeferencing using ground control points (GCPs), and calculating
displacement between a swath image and a reference image.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import dask.array as da
import geotiepoints as gtp
import numpy as np
import rioxarray
import xarray as xr
from numba import njit
from pyorbital.geoloc_avhrr import estimate_time_and_attitude_deviations
from pyproj import Geod
from pyresample import gradient
from pyresample.geometry import AreaDefinition, SwathDefinition
from rasterio.transform import xy
from rasterio.windows import from_bounds
from scipy.spatial import cKDTree

import georeferencer.displacement_calc as dc
import georeferencer.gcp_generation as gcp_gen

_geod = Geod(ellps="WGS84")
logger = logging.getLogger(__name__)
INVALID_DISPLACEMENT = (-100, -100)


def subset_from_bounds(buffer, dataset, max_lat, max_lon, min_lat, min_lon):
    """Selects a subset from a dataset based on latitude and longitude bounds."""
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
    width = int(subset_window.width / 2) * 2
    height = int(subset_window.height / 2) * 2

    subset = dataset.isel(x=slice(col_off, col_off + width), y=slice(row_off, row_off + height))
    return subset


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
    dataset = rioxarray.open_rasterio(filepath, engine="rasterio", chunks=512)
    subset = subset_from_bounds(buffer, dataset, max_lat, max_lon, min_lat, min_lon)

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
    gcp_x_original = gcp_x * 16 + 24 * 2
    gcp_y_original = gcp_y * 16 + 24 * 2

    xp, yp = xy(geo_transform, gcp_y_original, gcp_x_original)

    return xp, yp


def translate_gcp_to_swath_coordinates(gcp_array, calibrated_ds, geo_transform):
    """Convert GCP pixel coordinates into swath image coordinates based on latitude and longitude values.

    Args:
        gcp_array (array-like): Array of GCP (x, y) pixel coordinates.
        calibrated_ds (xarray.Dataset): Calibrated swath dataset containing longitude and latitude data.
        geo_transform (Affine): Geotransform matrix for coordinate conversion.

    Returns:
        (gcp coords, gcp lonlats): valid GCPs that fall within the swath image bounds, image coords and lonlats.
    """
    if "tc_longitude" in calibrated_ds and "tc_latitude" in calibrated_ds:
        lons = calibrated_ds["tc_longitude"].values
        lats = calibrated_ds["tc_latitude"].values
    else:
        lons = calibrated_ds["longitude"].values
        lats = calibrated_ds["latitude"].values

    image_shape = lons.shape
    lon_lat_pairs = np.column_stack((lons.ravel(), lats.ravel()))
    valid_mask = ~np.isnan(lon_lat_pairs).any(axis=1)
    valid_lon_lat_pairs = lon_lat_pairs[valid_mask]

    # TODO change to another KDTree
    source_geo_def = cKDTree(valid_lon_lat_pairs)

    min_lon = np.nanmin(lons)
    max_lon = np.nanmax(lons)
    min_lat = np.nanmin(lats)
    max_lat = np.nanmax(lats)

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
    swath_longitudes,
    swath_latitudes,
):
    """Reproject a reference image to match the swath image coordinate system.

    This function aligns a reference image with the swath data by using its
    geospatial bounds, regridding, and resampling it to match the swath.

    Args:
        matrix (np.array): Reference image data as a NumPy array.
        swath_longitudes (xarray.DataArray): Swath image longitudes.
        swath_latitudes (xarray.DataArray): Swath image latitudes.

    Returns:
        xarray.DataArray: The reprojected reference image aligned with the swath.
    """
    swath_lons_values = swath_longitudes.values
    swath_lats_values = swath_latitudes.values
    lon_min_swath = np.nanmin(swath_lons_values)
    lon_max_swath = np.nanmax(swath_lons_values)
    lat_min_swath = np.nanmin(swath_lats_values)
    lat_max_swath = np.nanmax(swath_lats_values)
    sub_matrix = subset_from_bounds(0, matrix, lat_max_swath, lon_max_swath, lat_min_swath, lon_min_swath)
    bounds = sub_matrix.rio.bounds()
    sub_matrix = sub_matrix.coarsen(x=2, y=2, boundary="trim").mean()
    source_area = AreaDefinition(
        "subgrid_area",
        "reference image AD around swath lat,lons",
        "tif_projection",
        sub_matrix.rio.crs,
        sub_matrix.shape[1],
        sub_matrix.shape[0],
        bounds,
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
    if "tc_longitude" in calibrated_ds and "tc_latitude" in calibrated_ds:
        lons = calibrated_ds["tc_longitude"]
        lats = calibrated_ds["tc_latitude"]
    else:
        lons = calibrated_ds["longitude"]
        lats = calibrated_ds["latitude"]

    tif = open_subset_tif(
        reference_image_path,
        np.min(lats),
        np.max(lats),
        np.min(lons),
        np.max(lons),
    )
    return tif.sel(band=1).astype(np.float32) / 2.55


def _reproject_to_swath(ref_image, calibrated_ds):
    """Reprojects the reference image into the swath coordinate system."""
    if "tc_longitude" in calibrated_ds and "tc_latitude" in calibrated_ds:
        lons = calibrated_ds["tc_longitude"]
        lats = calibrated_ds["tc_latitude"]
    else:
        lons = calibrated_ds["longitude"]
        lats = calibrated_ds["latitude"]
    return reproject_reference_into_swath(
        ref_image,
        lons,
        lats,
    ).astype(np.float32)


def _generate_gcps(ref_image):
    """Generates ground control point candidates based on variance analysis."""
    STEP = 8
    WINDOW_SIZE = 48
    THINNING_RADIUS = 11
    GROUP_SIZE = 3

    variance_array = gcp_gen.get_variance_array(
        ref_image.coarsen(x=2, y=2, boundary="trim").mean().values,
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
    return valid_gcps, valid_gcp_lonlats, valid_swath_coords


def get_swath_displacement(calibrated_ds, sun_zen, sat_zen, reference_image_path, dem_path=None):
    """Calculate the displacement between a swath image and a reference image.

    This function extracts a subset of the reference image, identifies
    ground control points (GCPs), and calculates displacement based on
    covariance analysis.

    Args:
        calibrated_ds (xarray.Dataset): Calibrated swath dataset containing channel data.
        sun_zen (2d Matrix): A 2d matrix with the same shape as the channel data containing the sun zenith angles.
        sat_zen (2d Matrix): A 2d matrix with the same shape as the channel data containing the satellite zenith angles.
        reference_image_path (str): Path to the reference GeoTIFF image.
        dem_path (str, optional): Path to the DEM GeoTIFF image.

    Returns:
        tuple: Time, attitude and distances between the swath and reference image as
            `(time difference, (roll, pitch, yaw), (distances_original, distances_minimized))`

    Raises:
        ValueError: If no valid displacement is found.
    """
    if dem_path:
        calibrated_ds = orthocorrection(calibrated_ds, sat_zen, dem_path)
    swath = _build_swath_image(calibrated_ds, sun_zen)
    ref_image = _load_reference_image(reference_image_path, calibrated_ds)
    geo_transform = ref_image.rio.transform()
    target_area = _reproject_to_swath(ref_image, calibrated_ds)
    gcp_points = _generate_gcps(ref_image)
    swath_coords, gcp_lonlats = translate_gcp_to_swath_coordinates(gcp_points, calibrated_ds, geo_transform)
    gcps, valid_gcp_lonlats, valid_swath_coords = _calculate_valid_gcps_from_swath_alignment(
        swath_coords, gcp_lonlats, swath, np.nan_to_num(target_area.values, nan=0.0)
    )

    valid_swath_coords = np.array(valid_swath_coords)
    valid_gcp_lonlats = np.array(valid_gcp_lonlats)
    gcps = np.array(gcps)
    y, x = valid_swath_coords[:, 0], valid_swath_coords[:, 1]
    lon, lat = valid_gcp_lonlats[:, 0], valid_gcp_lonlats[:, 1]
    y_corrected, x_corrected = gcps[:, 0], gcps[:, 1]
    y_displacement = valid_swath_coords[:, 0] - gcps[:, 0]
    x_displacement = valid_swath_coords[:, 1] - gcps[:, 1]

    calibrated_ds["gcp_x"] = xr.DataArray(x, dims=["points"])
    calibrated_ds["gcp_y"] = xr.DataArray(y, dims=["points"])
    calibrated_ds["gcp_longitude"] = xr.DataArray(lon, dims=["points"])
    calibrated_ds["gcp_latitude"] = xr.DataArray(lat, dims=["points"])
    calibrated_ds["gcp_x_corrected"] = xr.DataArray(x_corrected, dims=["points"])
    calibrated_ds["gcp_y_corrected"] = xr.DataArray(y_corrected, dims=["points"])
    calibrated_ds["gcp_x_displacement"] = xr.DataArray(x_displacement, dims=["points"])
    calibrated_ds["gcp_y_displacement"] = xr.DataArray(y_displacement, dims=["points"])

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


def get_swath_displacement_with_filename(swath_file, tle_dir, tle_file, reference_image_path, dem_path=None):
    """Compute swath displacement using a satellite swath file and a reference image.

    This function reads a swath file, retrieves the calibrated dataset,
    and calculates displacement relative to the reference image.

    Args:
        swath_file (str): Path to the swath data file.
        tle_dir (str): Directory containing Two-Line Elements (TLE) files.
        tle_file (str): Name of the specific TLE file to use.
        reference_image_path (str): Path to the reference GeoTIFF image.
        dem_path (str): Path to the DEM GeoTIFF image.

    Returns:
        tuple: Displacement values (dx, dy) between the swath and reference image.
    """
    from pygac import get_reader_class

    reader_cls = get_reader_class(swath_file)
    reader = reader_cls(tle_dir=tle_dir, tle_name=tle_file)
    reader.read(swath_file)
    calibrated_ds = reader.get_calibrated_dataset()
    _, sat_zen, _, sun_zen, _ = reader.get_angles()

    return get_swath_displacement(calibrated_ds, sun_zen, sat_zen, reference_image_path, dem_path)


@njit(nogil=True)
def _orthocorrect_line(elev, dists, tanz, swath_lats_i, swath_lons_i, lon_sup, lat_sup, out_lat, out_lon):
    npix = swath_lats_i.shape[0]
    nadir = np.argmin(tanz)
    sup_nadir = nadir * 4
    for j in range(npix):
        if tanz[j] == 0:
            out_lat[j] = swath_lats_i[j]
            out_lon[j] = swath_lons_i[j]
            continue
        idx_nom = j * 4
        best_idx = idx_nom
        if idx_nom <= sup_nadir:
            for k in range(idx_nom, sup_nadir):
                delta = abs(dists[k] - dists[idx_nom])
                if elev[k] > (delta / tanz[j]):
                    best_idx = k
        else:
            for k in range(idx_nom, sup_nadir, -1):
                delta = abs(dists[idx_nom] - dists[k])
                if elev[k] > (delta / tanz[j]):
                    best_idx = k

        out_lat[j] = lat_sup[best_idx]
        out_lon[j] = lon_sup[best_idx]


def _process_scanline(i, dem_swath, lons, lats, swath_lats, swath_lons, sat_zen):
    elev = dem_swath[i]
    lon_sup = lons[i]
    lat_sup = lats[i]

    # Calculate distances
    lon0, lat0 = lon_sup[:-1], lat_sup[:-1]
    lon1, lat1 = lon_sup[1:], lat_sup[1:]
    _, _, seg = _geod.inv(lon0, lat0, lon1, lat1)

    dists = np.zeros(len(seg) + 1, dtype=np.float64)
    dists[1:] = np.cumsum(seg)

    # Calculate tan of view angles
    tanz = np.tan(np.deg2rad(sat_zen[i]))

    out_lat_line = np.full_like(swath_lats[i], np.nan, dtype=np.float64)
    out_lon_line = np.full_like(swath_lons[i], np.nan, dtype=np.float64)

    _orthocorrect_line(elev, dists, tanz, swath_lats[i], swath_lons[i], lon_sup, lat_sup, out_lat_line, out_lon_line)

    return out_lat_line, out_lon_line


def _get_dem_swath(dem_file_path, min_lats, max_lats, min_lon, max_lon, lons_sup, lats_sup):
    src = open_subset_tif(dem_file_path, min_lats, max_lats, min_lon, max_lon).squeeze(drop=True)

    transform = src.rio.transform()
    x_min, y_max, x_res, y_res = transform.c, transform.f, transform.a, transform.e
    nrows, ncols = src.sizes["y"], src.sizes["x"]

    cols = np.round((lons_sup - x_min) / x_res).astype(int)
    rows = np.round((lats_sup - y_max) / y_res).astype(int)

    valid_mask = (rows >= 0) & (rows < nrows) & (cols >= 0) & (cols < ncols)

    rows_flat = rows.ravel()[valid_mask.ravel()]
    cols_flat = cols.ravel()[valid_mask.ravel()]

    values = src.data.vindex[(rows_flat, cols_flat)].compute(scheduler="threads")

    output = np.full(lons_sup.size, np.nan)
    output[np.where(valid_mask.ravel())[0]] = values

    return output.reshape(lons_sup.shape)


def orthocorrection(calibrated_ds, sat_zen, dem_file_path):
    """Performs orthocorrection on latitude and longitude based on satellite viewing angles."""
    swath_lons = calibrated_ds["longitude"]
    swath_lats = calibrated_ds["latitude"]
    lons, lats = lonlat_interpolator(swath_lons, swath_lats, np.arange(2048), np.arange(0, 2048, 0.25))

    dem_swath = _get_dem_swath(
        dem_file_path, np.min(swath_lats), np.max(swath_lats), np.min(swath_lons), np.max(swath_lons), lons, lats
    )
    out_lats = np.full_like(swath_lats, np.nan, dtype=np.float64)
    out_lons = np.full_like(swath_lons, np.nan, dtype=np.float64)
    nscan, _ = swath_lons.shape
    process_func = partial(
        _process_scanline,
        dem_swath=dem_swath,
        lons=lons,
        lats=lats,
        swath_lats=swath_lats.values,
        swath_lons=swath_lons.values,
        sat_zen=sat_zen,
    )

    num_workers = int(os.getenv("THREADS", os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(process_func, range(nscan))

        for i, (lat_line, lon_line) in enumerate(results):
            out_lats[i] = lat_line
            out_lons[i] = lon_line

    calibrated_ds["tc_latitude"] = (calibrated_ds["latitude"].dims, out_lats)
    calibrated_ds["tc_longitude"] = (calibrated_ds["longitude"].dims, out_lons)
    return calibrated_ds


def lonlat_interpolator(lons, lats, cols_subset, cols_full):
    """Interpolate from lat-lon tie-points to pixel locations."""
    rows = np.arange(len(lats))
    along_track_order = 1
    cross_track_order = 3
    satint = gtp.SatelliteInterpolator(
        (lons, lats), (rows, cols_subset), (rows, cols_full), along_track_order, cross_track_order
    )
    return satint.interpolate()


# -----------------------------------------------------------------------------
# Pure-Python implementations (for testing)
# -----------------------------------------------------------------------------
def _orthocorrect_line_py(elev, dists, tanz, swath_lats_i, swath_lons_i, lon_sup, lat_sup, out_lat, out_lon):
    npix = swath_lats_i.shape[0]
    nadir = npix // 2
    sup_nadir = nadir * 4
    for j in range(npix):
        if tanz[j] == 0:
            out_lat[j] = swath_lats_i[j]
            out_lon[j] = swath_lons_i[j]
            continue
        idx_nom = j * 4
        best_idx = idx_nom
        if idx_nom <= sup_nadir:
            for k in range(idx_nom, sup_nadir):
                delta = abs(dists[k] - dists[idx_nom])
                if elev[k] > (delta / tanz[j]):
                    best_idx = k
        else:
            for k in range(idx_nom, sup_nadir, -1):
                delta = abs(dists[idx_nom] - dists[k])
                if elev[k] > (delta / tanz[j]):
                    best_idx = k

        out_lat[j] = lat_sup[best_idx]
        out_lon[j] = lon_sup[best_idx]


# -----------------------------------------------------------------------------
# Pure-Python implementations (for testing)
# -----------------------------------------------------------------------------
def _process_scanline_py(i, dem_swath, lons, lats, swath_lats, swath_lons, sat_zen):
    elev = dem_swath[i]
    lon_sup = lons[i]
    lat_sup = lats[i]

    # Calculate distances
    lon0, lat0 = lon_sup[:-1], lat_sup[:-1]
    lon1, lat1 = lon_sup[1:], lat_sup[1:]
    _, _, seg = _geod.inv(lon0, lat0, lon1, lat1)

    dists = np.zeros(len(seg) + 1, dtype=np.float64)
    dists[1:] = np.cumsum(seg)

    # Calculate tan of view angles
    tanz = np.tan(np.deg2rad(sat_zen[i]))

    out_lat_line = np.full_like(swath_lats[i], np.nan, dtype=np.float64)
    out_lon_line = np.full_like(swath_lons[i], np.nan, dtype=np.float64)

    _orthocorrect_line_py(elev, dists, tanz, swath_lats[i], swath_lons[i], lon_sup, lat_sup, out_lat_line, out_lon_line)

    return out_lat_line, out_lon_line
