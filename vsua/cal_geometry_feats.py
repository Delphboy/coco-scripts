from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import pickle
from multiprocessing import Pool, Value

import argparse

import logging

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Counter(object):
    def __init__(self):
        self.val = Value('i', 0)

    def add(self, n=1):
        with self.val.get_lock():
            self.val.value += n

    @property
    def value(self):
        return self.val.value


def get_cwh(box):
    x_min, y_min, x_max, y_max = box
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.
    return cx, cy, w, h



def cal_geometry_feats(id):
    counter.add(1)
    info = box_info[id]
    boxes = info['boxes']
    num_boxes = boxes.shape[0]
    w, h = info['image_w'], info['image_h']
    scale = w * h
    diag_len = math.sqrt(w ** 2 + h ** 2)
    feats = np.zeros([num_boxes, num_boxes, num_features], dtype='float')
    for i in range(num_boxes):
        if is_directed:
            start = 0
        else:
            start = i
        for j in range(start, num_boxes):
            if is_directed:
                _box1, _box2 = boxes[i], boxes[j]
                _w1 = (_box1[2] - _box1[0]) + 1.
                _h1 = (_box1[3] - _box1[1]) + 1.
                _w2 = (_box2[2] - _box2[0]) + 1.
                _h2 = (_box2[3] - _box2[1]) + 1.
                # the larger box as box1, the other one as box2
                if (_w1*_h1)/(_w2*_h2) > 1:
                    box1, box2 = _box1, _box2
                else:
                    box2, box1 = _box1, _box2
            else:
                box1, box2 = boxes[i], boxes[j]

            cx1, cy1, w1, h1 = get_cwh(box1)
            cx2, cy2, w2, h2 = get_cwh(box2)
            x_min1, y_min1, x_max1, y_max1 = box1
            x_min2, y_min2, x_max2, y_max2 = box2
            # scale
            scale1 = w1 * h1
            scale2 = w2 * h2
            # Offset
            offsetx = cx2 - cx1
            offsety = cy2 - cy1
            # Aspect ratio
            aspect1 = w1 / h1
            aspect2 = w2 / h2
            # Overlap
            i_xmin = max(x_min1, x_min2)
            i_ymin = max(y_min1, y_min2)
            i_xmax = min(x_max1, x_max2)
            i_ymax = min(y_max1, y_max2)
            iw = max(i_xmax - i_xmin + 1, 0)
            ih = max(i_ymax - i_ymin + 1, 0)
            areaI = iw * ih
            areaU = scale1 + scale2 - areaI
            # dist
            dist = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
            # angle
            angle = math.atan2(cy2 - cy1, cx2 - cx1)

            f1 = offsetx / math.sqrt(scale1)
            f2 = offsety / math.sqrt(scale1)
            f3 = math.sqrt(scale2 / scale1)
            f4 = areaI / areaU
            f5 = aspect1
            f6 = aspect2
            f7 = dist / diag_len
            f8 = angle
            feat = [f1, f2, f3, f4, f5, f6, f7, f8]
            feats[i][j] = np.array(feat)
    if counter.value % 100 == 0 and counter.value >= 100:
        # print('{} / {}'.format(counter.value, num_images))
        logger.info('{} / {}'.format(counter.value, num_images))
    return id, feats


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--num_feats', type=int, default=8)
    args.add_argument('--directed', type=bool, default=False)
    args.add_argument("--input_file", type=str, help='Path to vsua_box_info.pkl')
    args.add_argument('--save_path', type=str, help='Path to save geometry features')

    args = args.parse_args()

    num_features = args.num_feats
    is_directed = args.directed
    save_path = args.save_path + "-{}directed.pkl".format("" if is_directed else "un")

    with open(args.input_file, 'rb') as f:
        box_info = pickle.load(f, encoding='latin1')

    num_images = len(box_info)
    counter = Counter()

    p = Pool(20)
    logger.info("[INFO] Start")
    results = p.map(cal_geometry_feats, box_info.keys())
    all_feats = {res[0]: res[1] for res in results}
    logger.info("[INFO] Finally %d processed" % len(all_feats))

    with open(save_path, 'wb') as f:
        pickle.dump(all_feats, f)
    print("saved")