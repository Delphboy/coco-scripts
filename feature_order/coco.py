import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from PIL import Image

class CocoButdFeatures(Dataset):
    def __init__(
        self,
        captions_file: str,
        image_dir: str,
        feature_dir: str,
        bbox_dir: str,
    ):
        self.captions_file = captions_file
        self.image_dir = image_dir
        self.feature_dir = feature_dir
        self.bbox_dir = bbox_dir

        with open(self.captions_file, "r") as f:
            self.captions_file_data = json.load(f)

        self.image_locations = []
        self.feature_locations = []
        self.box_locations = []

        for image_data in self.captions_file_data["images"]:
            self.image_locations.append(
                os.path.join(
                    self.image_dir,
                    image_data["filepath"],
                    image_data["filename"],
                )
            )

            feat_path = os.path.join(
                    self.feature_dir,
                    (f"{image_data['cocoid']}.npz"),
                )
            self.feature_locations.append(feat_path)

            bbox_path = os.path.join(
                self.bbox_dir,
                (f"{image_data['cocoid']}.npy"),
            )
            self.box_locations.append(bbox_path)

    def __getitem__(self, index):
        image = Image.open(self.image_locations[index])
        image = np.array(image)

        features = np.load(self.feature_locations[index])["feat"]
        
        box_loc = self.box_locations[index]
        boxes = np.load(box_loc)
        boxes = torch.from_numpy(boxes).type(torch.float32)

        name = box_loc.split("/")[-1].split(".")[0]

        return name, image, features, boxes

    def __len__(self):
        return len(self.box_locations)
