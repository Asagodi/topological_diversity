"""Script to make .mp4 files from .png plots with the same prefix.

.mp4 files are made so that the frames-per-second can be controlled.
"""
import imageio.v3 as iio
import os
import sys
import re
import numpy as np

folder = sys.argv[1]
folder = os.path.abspath(folder)
print("Making gifs from plots in folder: ", folder)

pattern = re.compile(r'_iter\d+')
def get_iteration_span(fname):
    match = pattern.search(fname)
    if match:
        return match.span()
    return None

def get_iteration(fname):
    span = get_iteration_span(fname)
    if span:
        return int(fname[span[0]+len('_iter'):span[1]])
    return -1

def get_prefix(fname):
    """Gets the part before the iteration number."""
    span = get_iteration_span(fname)
    if span:
        return fname[:span[0]]
    return None

def get_all_prefixes(folder):
    all_prefixes = set()
    for fname in os.listdir(folder):
        prefix = get_prefix(fname)
        if prefix:
            all_prefixes.add(prefix)
    return all_prefixes

if __name__ == '__main__':
    for prefix in get_all_prefixes(folder):
        print(f"Making gif of {prefix} images...")
        image_names = [
            os.path.join(folder,f)
            for f in os.listdir(folder)
            if f.lower().endswith('.png')
            and f.startswith(prefix)]
        image_names = sorted(
            image_names,
            key=get_iteration)
        images = np.stack([
            iio.imread(f) for f in image_names])
        mp4_fname = os.path.join(folder, prefix+".mp4")
        iio.imwrite(mp4_fname, images, fps=2, plugin='FFMPEG')
    
