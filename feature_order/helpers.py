import numpy as np
import os
from multiprocessing import Pool

def load_feature(feature_dir):
    return np.load(feature_dir)["feat"]


def load_bbox(bbox_dir):
    return np.load(bbox_dir)  # [B, 4]


def save_feature(output_dir, name, features):
    path = os.path.join(output_dir, f"{name}.npz")
    np.savez_compressed(path, feat=features)


def process_parallel(feature_dir, bbox_dir, output_dir, function):
    files = os.listdir(feature_dir)
    with Pool() as p:
        p.starmap(
            function,
            [(feature_dir, bbox_dir, output_dir, file) for file in files],
        )
