import rioxarray
import dask.array as da
import numpy as np
import gcp_generation as gcp_gen
from pyresample.geometry import AreaDefinition, SwathDefinition
from pyresample import gradient
from scipy.spatial import cKDTree
from rasterio.windows import from_bounds
from rasterio.transform import xy

import discalc

def open_subset_tif(filepath, min_lat, max_lat, min_lon, max_lon):
    """Open a subset of a GeoTIFF file based on latitude and longitude bounds."""
    
    # Open the dataset lazily using rioxarray (Dask-backed)
    dataset = rioxarray.open_rasterio(filepath, engine="rasterio")
    
    # Get the transform and CRS for the dataset
    transform = dataset.rio.transform()
    crs = dataset.rio.crs
    
    # Calculate the window based on bounding box
    subset_window = from_bounds(
        left=min_lon, bottom=min_lat, right=max_lon, top=max_lat, transform=transform
    )

    # Convert window attributes to integer offsets and dimensions
    col_off = int(subset_window.col_off)
    row_off = int(subset_window.row_off)
    width = int(subset_window.width)
    height = int(subset_window.height)

    # Subset the dataset lazily (preserve Dask-backed DataArray)
    subset = dataset.isel(
        x=slice(col_off, col_off + width), 
        y=slice(row_off, row_off + height)
    )

    # Return the subset as a Dask-backed xarray.DataArray
    return subset

def gcp_to_lonlat(gcp_x, gcp_y, geo_transform):
    """
    Convert GCP pixel coordinates to geographic coordinates (lat/lon).
    """
    #TODO what is the magic behind 2.5......
    gcp_x_original = (gcp_x + 2.5) * 16
    gcp_y_original = (gcp_y + 2.5) * 16

    xp, yp = xy(geo_transform, gcp_y_original, gcp_x_original)

    return xp, yp

def translate_gcp_to_swath_coordinates(gcp_array, swath_ds=None, geo_transform=None):
    """
    Translates GCP pixel coordinates to swath image coordinates based on lat/lon values.
    
    :param gcp_array: Array of GCP pixel (x, y) coordinates
    :param dataset: GDAL dataset (GeoTIFF or similar)
    :param swath_ds: Dataset containing swath latitude and longitude arrays
    :return: List of legit GCPs within the swath bounds
    """
    longitudes = swath_ds.attrs['area'].lons.values
    latitudes = swath_ds.attrs['area'].lats.values
    image_shape = longitudes.shape

    lon_lat_pairs = np.column_stack((longitudes.ravel(), latitudes.ravel()))
    valid_mask = ~np.isnan(lon_lat_pairs).any(axis=1)
    valid_lon_lat_pairs = lon_lat_pairs[valid_mask]

    #TODO change to another KDTree
    source_geo_def = cKDTree(valid_lon_lat_pairs)

    min_lon = np.nanmin(longitudes)
    max_lon = np.nanmax(longitudes)
    min_lat = np.nanmin(latitudes)
    max_lat = np.nanmax(latitudes)

    nr_of_valid_gcps = 0
    swath_cords = []
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

    print(f"Number of valid GCPs: {nr_of_valid_gcps}")
    return swath_cords

def reproject_reference_into_swath(matrix, crs, swath_longitudes, swath_latitudes, origin_x, origin_y, \
                                   pixel_size_x, pixel_size_y, min_x, min_y, max_x, max_y):
        
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

        sub_matrix = np.array(gcp_gen.downsample_2x2_to_1km(sub_matrix), \
                                      dtype=np.float32)
        sub_matrix = da.from_array(sub_matrix, chunks=(1000, 1000))

        area_id = 'subgrid_area'
        description = f'reference image AD around swath lat,lons'
        proj_id = 'tif_projection'
        proj_dict = crs

        source_area = AreaDefinition(
            area_id, description, proj_id, proj_dict,
            sub_matrix.shape[1], sub_matrix.shape[0],
            [lon_min_source, lat_min_source, lon_max_source, lat_max_source]
        )
        res = gradient.create_gradient_search_resampler(source_area, SwathDefinition(lons=swath_longitudes,\
                                                                                           lats=swath_latitudes))
        resampled_data=res.resample(sub_matrix)
        return resampled_data

print("hello")