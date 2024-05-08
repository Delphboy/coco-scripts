import os
import argparse
import numpy as np

from helpers import load_feature, load_bbox, save_feature, process_parallel

def caculate_area(bbox):
    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    x2 = bbox[:, 2]
    y2 = bbox[:, 3]
    area = (x2 - x1) * (y2 - y1)
    return area

def sort_feature_by_area(features, bbox):
    area = caculate_area(bbox)
    sort_index = np.argsort(area)[::-1]
    features = features[sort_index]
    return features

def process_individual(feature_dir, bbox_dir, output_dir, file):
    feature_path = os.path.join(feature_dir, file)
    features = load_feature(feature_path)

    bbox_file = file.split(".")[0] + ".npy"
    bbox_path = os.path.join(bbox_dir, bbox_file)
    bbox = load_bbox(bbox_path)

    sorted_features = sort_feature_by_area(features, bbox)
    save_feature(output_dir, file.split(".")[0], sorted_features)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--feature_dir", type=str, required=True)
    args.add_argument("--output_dir", type=str, required=True)
    args.add_argument("--bbox_dir", type=str, required=True)

    args = args.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    process_parallel(args.feature_dir, args.bbox_dir, args.output_dir, process_individual)
