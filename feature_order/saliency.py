import argparse
from multiprocessing import Pool
import os

import cv2
import numpy as np

from coco import CocoButdFeatures
from helpers import save_feature


def get_order_index_from_saliency(image, bboxes):
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    saliency_scores = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        saliency_score = np.mean(saliencyMap[int(y1):int(y2), int(x1):int(x2)])
        saliency_scores.append(saliency_score)

    return np.argsort(saliency_scores)[::-1]


def process_individual(dataset, index, output_dir):
    name, image, features, boxes = dataset.__getitem__(index)
    order = get_order_index_from_saliency(image, boxes)
    ordered_features = features[order]
    save_feature(output_dir, name, ordered_features)


def process_parallel(output_dir, dataset, function):
    with Pool() as p:
        p.starmap(function, [(dataset, i, output_dir) for i in range(len(dataset))])


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--karpathy_json_file", type=str, required=True)
    args.add_argument("--image_dir", type=str, required=True)
    args.add_argument("--feature_dir", type=str, required=True)
    args.add_argument("--output_dir", type=str, required=True)
    args.add_argument("--bbox_dir", type=str, required=True)

    args = args.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = CocoButdFeatures(
        captions_file=args.karpathy_json_file,
        image_dir=args.image_dir,
        feature_dir=args.feature_dir,
        bbox_dir=args.bbox_dir,
    )

    process_parallel(args.output_dir, dataset, process_individual)
