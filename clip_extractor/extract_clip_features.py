import argparse
import torch
import clip
import numpy as np
import os
from coco import CocoButdFeatures
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=DEVICE)


def save_clip_features(output_dir, features):
    feats = {"feat": features}
    np.savez_compressed(output_dir, **feats)


def extract_clip_features(image, boxes):
    # Get pixels for each box
    box_features = []
    for box in boxes:
        box = box.type(torch.int32)
        box = box.tolist()
        box = [int(x) for x in box]
        box = tuple(box)
        box_image = image.crop(box)
        box_image = preprocess(box_image).unsqueeze(0).to(DEVICE)
        box_features.append(model.encode_image(box_image))
    box_features = torch.stack(box_features, dim=0)
    return box_features.squeeze(1).cpu().detach().numpy()

def process(i):
    logger.info(f"Processing {i+1}/{len(dataset)}")
    name, boxes, image = dataset.__getitem__(i)
    clip_feats = extract_clip_features(image, boxes)
    save_dir = os.path.join(args.output_dir, f"{name}.npz")
    save_clip_features(save_dir, clip_feats)

if __name__ == "__main__":
    global dataset
    parser = argparse.ArgumentParser(description="Extract CLIP features for BUTD detected objects")
    parser.add_argument("--karpathy_json_file", type=str, required=True, help="Path to the Karpathy JSON file")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the image root directory")
    parser.add_argument("--butd_box_dir", type=str, required=True, help="Path to the BUTD bounding box location")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the extracted features")

    args = parser.parse_args()
    logger.info(f"Device set to: {DEVICE}")
    dataset = CocoButdFeatures(args.karpathy_json_file, args.butd_box_dir, args.image_dir)
    logger.info("Dataset loaded")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(len(dataset)):
        process(i)
