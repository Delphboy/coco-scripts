import json
import os
from itertools import chain

import numpy as np
import torch
from torch.utils.data import Dataset

# import PIL
from PIL import Image

class CocoButdFeatures(Dataset):
    def __init__(
        self,
        captions_file: str,
        feature_dir: str,
        image_dir: str,
    ):
        self.captions_file = captions_file
        self.feature_dir = feature_dir
        self.image_dir = image_dir

        with open(self.captions_file, "r") as f:
            self.captions_file_data = json.load(f)

        self.box_locations = []
        self.image_locations = []

        for image_data in self.captions_file_data["images"]:
            feat_path = os.path.join(
                self.feature_dir,
                (f"{image_data['cocoid']}.npy"),
            )
            self.box_locations.append(feat_path)
            self.image_locations.append(
                os.path.join(
                    self.image_dir,
                    image_data["filepath"],
                    image_data["filename"],
                )
            )

    def __getitem__(self, index):
        box_loc = self.box_locations[index]
        name = box_loc.split("/")[-1].split(".")[0]
        boxes = np.load(box_loc)
        boxes = torch.from_numpy(boxes).type(torch.float32)

        image = Image.open(self.image_locations[index])

        return name, boxes, image

    def __len__(self):
        return len(self.box_locations)
