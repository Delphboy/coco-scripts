import os
import shutil
import numpy as np

butd_root = "/data/EECS-YuanLab/COCO/butd_att/"
geo_graph_root = "/data/EECS-YuanLab/COCO/geometry-iou-iou0.2-dist0.5-undirected/"
geo_graph_bkp_root = os.path.join(geo_graph_root, "bkp")


def load_butd_file(dir):
    return np.load(dir, allow_pickle=True, encoding="latin1")["feat"]


def load_geo_file(dir):
    return np.load(dir, allow_pickle=True, encoding="latin1").item()


def check_specific_files(id: int):
    butd = load_butd_file(os.path.join(butd_root, str(id) + ".npz"))
    geog = load_geo_file(os.path.join(geo_graph_root, str(id) + ".npy"))

    butd_count = butd.shape[0]
    max_edge_idx = geog["edges"].reshape(-1).max()

    print(f"File {id} has {butd_count} nodes but a max edge index of {max_edge_idx}")


def get_geo_file_from_butd_file(butd_file: str) -> str:
    _, butd_file_name = os.path.split(butd_file)
    file_id = butd_file_name.split(".")[0]
    geo_graph_file = os.path.join(geo_graph_root, file_id + ".npy")
    return geo_graph_file


def check_geo_edge_idxs_do_not_exceed_butd_feat_dims():
    edge_cases = []
    for butd_file in os.listdir(butd_root):
        geo_graph_file = get_geo_file_from_butd_file(butd_file)
        butd_file = os.path.join(butd_root, butd_file)
        assert os.path.isfile(butd_file)
        assert os.path.isfile(geo_graph_file)

        print(f"Processing {os.path.split(butd_file)[1]}")
        butd = load_butd_file(butd_file)
        geog = load_geo_file(geo_graph_file)

        butd_count = butd.shape[0]
        max_edge_idx = geog["edges"].reshape(-1).max()
        assert max_edge_idx <= butd_count, "There is a bad edge"
        if max_edge_idx == butd_count:
            edge_cases.append(butd_file)

        geog_edge_count = geog["edges"].shape[0]
        geog_edge_feat_count = geog["feats"].shape[0]
        assert (
            geog_edge_count == geog_edge_feat_count
        ), "Different number of edges and edge features"
    print(f"There are {len(edge_cases)} edge cases")
    print(edge_cases)
    return edge_cases


if __name__ == "__main__":
    os.makedirs(geo_graph_bkp_root, exist_ok=True)
    edge_cases = check_geo_edge_idxs_do_not_exceed_butd_feat_dims()
    for edge_case in edge_cases:
        # Load file data
        geo_file = get_geo_file_from_butd_file(edge_case)
        geo_data = load_geo_file(geo_file)
        edges = geo_data["edges"]
        edge_feats = geo_data["feats"]

        max_allowed_edge_index = load_butd_file(
            os.path.join(butd_root, edge_case)
        ).shape[0]

        # Remove bad edges
        bad_indices = np.where(edges >= max_allowed_edge_index)
        new_edges = np.delete(edges, bad_indices[0], axis=0)
        new_feats = np.delete(edge_feats, bad_indices[0], axis=0)

        print(edges.shape, new_edges.shape, edge_feats.shape, new_feats.shape)

        # Move old file to bkp
        geo_file_name = os.path.split(geo_file)[1]
        shutil.copy(geo_file, os.path.join(geo_graph_bkp_root, geo_file_name))

        # Overwrite edge data
        new_geo_data = {"edges": new_edges, "feats": new_feats}
        print(geo_file)
        np.save(geo_file, new_geo_data)
        print(f"Fixed {edge_case}")

