# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import argparse
import os
import pickle
from multiprocessing import Pool, Value

import numpy as np

"""
build geometry graph & obtain its edges and edge features
assign an edge between two boxes if their iou and distance satisfy given threshold
"""


class Counter(object):
    def __init__(self):
        self.val = Value("i", 0)

    def add(self, n=1):
        with self.val.get_lock():
            self.val.value += n

    @property
    def value(self):
        return self.val.value


def build_geometry_graph(id):
    feats = all_feats[id]
    num_boxes = feats.shape[0]
    edges = []
    relas = []
    for i in range(num_boxes):
        if is_directed:
            start = 0
        else:
            start = i
        for j in range(start, num_boxes):
            if i == j:
                continue
            # iou and dist thresholds
            if feats[i][j][3] < iou or feats[i][j][6] > dist:
                continue
            edges.append([i, j])
            relas.append(feats[i][j])

    # in case some trouble is met
    if edges == []:
        edges.append([0, 1])
        relas.append(feats[0][1])

    edges = np.array(edges)
    relas = np.array(relas)
    graph = {}
    graph["edges"] = edges
    graph["feats"] = relas
    np.save(os.path.join(save_dir, str(id)), graph)

    if counter.value % 100 == 0 and counter.value >= 100:
        print("{} / {}".format(counter.value, num_images))


args = argparse.ArgumentParser()
args.add_argument('--directed', type=bool, default=False)
args.add_argument('--iou', type=float, default=0.2)
args.add_argument("--dist", type=float, default=0.5)
args.add_argument('--save_path', type=str)
args.add_argument('--geometry_path', type=str)

args = args.parse_args()

is_directed = args.directed  # directed or undirected graph
iou = 0.2
dist = 0.5

save_dir = args.save_path + "-iou{}-dist{}-{}directed".format(
    iou, dist, "" if is_directed else "un"
)

geometry_feature_path = args.geometry_path + "-{}directed.pkl".format("" if is_directed else "un")

if os.path.exists(save_dir):
    raise Exception("dir already exists")
else:
    os.makedirs(save_dir, exist_ok=True)

counter = Counter()
print("loading geometry features of all box pairs....")
with open(geometry_feature_path, "rb") as f:
    all_feats = pickle.load(f)
num_images = len(all_feats)
print("Loaded %d images...." % num_images)


p = Pool(20)
print("[INFO] Start")
results = p.map(build_geometry_graph, all_feats.keys())
print("Done")