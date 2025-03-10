"""Script for running georeferencer from command line."""

import argparse
import logging

import georeferencer.georeferencer as gr

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser("Find displacement between a .l1b image and a reference image")
parser.add_argument("--filename", type=str, required=True, help="Path to a .l1b file, including name of the file")
parser.add_argument("--tle_dir", type=str, required=True, help="Path to a directory where tle files are located")
parser.add_argument("--tle_file", type=str, required=True, help="Name of tle file")
parser.add_argument(
    "--reference",
    type=str,
    required=True,
    help="Path to a .tif file that acts as a reference image,\
                     including name of the file",
)

args = parser.parse_args()

logging.info("Finding displacement between %s and the reference image %s", args.filename, args.reference)
displacement = gr.get_swath_displacement_with_filename(args.filename, args.tle_dir, args.tle_file, args.reference)
logging.info("Result: %s", displacement)
