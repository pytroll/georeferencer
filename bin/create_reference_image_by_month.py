"""Script for creating reference images for georeferencer."""

import logging
import subprocess
import sys
from pathlib import Path

import cv2
import requests

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TILES = {
    "A1": (90.0, -180.0),
    "B1": (90.0, -90.0),
    "C1": (90.0, 0.0),
    "D1": (90.0, 90.0),
    "A2": (0.0, -180.0),
    "B2": (0.0, -90.0),
    "C2": (0.0, 0.0),
    "D2": (0.0, 90.0),
}

BASE_URL = "https://neo.gsfc.nasa.gov/archive/bluemarble/bmng/world_500m"
OCEANMASK_TILES_URL = "https://neo.gsfc.nasa.gov/archive/bluemarble/bmng/landmask/world.oceanmask.21600x21600."
OCEANMASK_EXTENSION = ".png"

RESOLUTION = 1 / 240


def download_file(url, path):
    """Downloads file."""
    if not path.exists():
        logger.info(f"Downloading {url}...")
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(path, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
            logger.info(f"Downloaded to {path}")
        else:
            raise RuntimeError(f"Failed to download: {url} — HTTP {r.status_code}")
    else:
        logger.info(f"Already exists: {path}")


def download_tile_png(month, tile, dest_dir):
    """Downloads tile for specific month and tile number."""
    filename = f"world.topo.2004{month:02d}.3x21600x21600.{tile}.png"
    url = f"{BASE_URL}/{filename}"
    path = dest_dir / filename
    download_file(url, path)
    return path


def download_tile_ocean_mask(tile, dest_dir):
    """Downloads ocean mask for specific month and tile."""
    filename = f"world.oceanmask.21600x21600.{tile}.png"
    url = f"{OCEANMASK_TILES_URL}{tile}{OCEANMASK_EXTENSION}"
    path = dest_dir / filename
    download_file(url, path)
    return path


def apply_mask_and_convert_to_tif(tile_png, ocean_mask_png, output_tif, tile_key):
    """Applies ocean mask png to tile png."""
    if Path(output_tif).exists():
        logger.info(f"{output_tif} already exists. Skipping.")
        return

    logger.info(f"Masking and converting {tile_png} to {output_tif}...")

    img = cv2.imread(str(tile_png))
    mask = cv2.imread(str(ocean_mask_png), cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        raise ValueError(f"Failed to load image or mask for tile {tile_png}")

    masked = cv2.bitwise_and(img, img, mask=mask)

    masked_png = Path(output_tif).with_suffix(".masked.png")
    cv2.imwrite(str(masked_png), masked)

    ul_lat, ul_lon = TILES[tile_key]

    cmd = [
        "gdal_translate",
        "-of",
        "GTiff",
        "-a_ullr",
        str(ul_lon),
        str(ul_lat),
        str(ul_lon + 90),
        str(ul_lat - 90),
        "-a_srs",
        "EPSG:4326",
        "-co",
        "BIGTIFF=YES",
        str(masked_png),
        str(output_tif),
    ]

    subprocess.run(cmd, check=True)
    masked_png.unlink()

    logger.info(f"Saved masked GeoTIFF to {output_tif}")


def merge_tiles(tiles, output_file):
    """Merge geotiff files."""
    logger.info("Merging tiles with GDAL Merge (gdal_merge.py)...")

    cmd = [
        "gdal_merge.py",
        "-o",
        output_file,
        "-of",
        "GTiff",
        "-co",
        "COMPRESS=ZSTD",
        "-co",
        "ZSTD_LEVEL=6",
        "-co",
        "TILED=YES",
        "-co",
        "BIGTIFF=YES",
    ] + [str(tile) for tile in tiles]

    subprocess.run(cmd, check=True)
    logger.info(f"Merged output: {output_file}")


def create_tif(month):
    """Create geotiff file."""
    dest_dir = Path("bmng_tiles")
    dest_dir.mkdir(exist_ok=True)

    masked_tifs = []

    for tile in TILES:
        tile_png = download_tile_png(month, tile, dest_dir)
        mask_png = download_tile_ocean_mask(tile, dest_dir)

        masked_tif = dest_dir / f"{tile}_{month}_masked.tif"
        apply_mask_and_convert_to_tif(tile_png, mask_png, masked_tif, tile)

        masked_tifs.append(masked_tif)

    output_file = f"geo_reference_2004{month:02d}.tif"
    merge_tiles(masked_tifs, output_file)
    return output_file


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python create_reference_image_by_month.py <month (1–12)>")
        sys.exit(1)

    try:
        month = int(sys.argv[1])
        if not (1 <= month <= 12):
            raise ValueError
    except ValueError:
        logger.error("Please provide a valid month number between 1 and 12.")
        sys.exit(1)

    merged_tif = create_tif(month)
    logger.info(f"Final merged and masked TIFF saved to {merged_tif}")
