import os
import numpy as np


butd_root = "/home/henry/Datasets/COCO/butd_att/"
geo_graph_root = "/home/henry/Datasets/COCO/geometry-iou-iou0.2-dist0.5-undirected/"


def load_butd_file(dir):
    return np.load(dir, allow_pickle=True, encoding="latin1")["feat"]


def load_geo_file(dir):
    return np.load(dir, allow_pickle=True, encoding="latin1").item()


def load_specific_files(id: int):
    butd = load_butd_file(os.path.join(butd_root, str(id) + ".npz"))
    geog = load_geo_file(os.path.join(geo_graph_root, str(id) + ".npy"))

    butd_count = butd.shape[0]
    max_edge_idx = geog["edges"].reshape(-1).max()

    print(f"File {id} has {butd_count} nodes but a max edge index of {max_edge_idx}")


def check_geo_edge_idxs_do_not_exceed_butd_feat_dims():
    edge_cases = []
    for butd_file in os.listdir(butd_root):
        _, butd_file_name = os.path.split(butd_file)
        file_id = butd_file_name.split(".")[0]
        geo_graph_file = os.path.join(geo_graph_root, file_id + ".npy")
        butd_file = os.path.join(butd_root, butd_file)
        assert os.path.isfile(butd_file)
        assert os.path.isfile(geo_graph_file)

        print(f"Processing {butd_file_name}")
        butd = load_butd_file(butd_file)
        geog = load_geo_file(geo_graph_file)

        butd_count = butd.shape[0]
        max_edge_idx = geog["edges"].reshape(-1).max()
        # assert max_edge_idx < butd_count, "There is a bad edge"
        assert max_edge_idx <= butd_count, "There is a bad edge"
        if max_edge_idx >= butd_count:
            edge_cases.append(butd_file)

        geog_edge_count = geog["edges"].shape[0]
        geog_edge_feat_count = geog["feats"].shape[0]
        assert (
            geog_edge_count == geog_edge_feat_count
        ), "Different number of edges and edge features"
    print(f"There are {len(edge_cases)} edge cases")
    print(edge_cases)


if __name__ == "__main__":
    # check_geo_edge_idxs_do_not_exceed_butd_feat_dims()
    load_specific_files(466838)
    load_specific_files(231631)
    print(f"Processed 2 * {len(os.listdir(butd_root))} files")
