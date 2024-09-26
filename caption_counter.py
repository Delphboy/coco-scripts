import argparse
import json
import os


def extract_captions(coco_json_file):
    train_caps = []
    val_caps = []
    test_caps = []
    with open(coco_json_file, "r") as f:
        coco = json.load(f)

    for img in coco["images"][:100]:
        if img["split"] == "train" or img["split"] == "restval":
            [train_caps.append(x["raw"]) for x in img["sentences"]]
        elif img["split"] == "val":
            [val_caps.append(x["raw"]) for x in img["sentences"]]
        else:
            [test_caps.append(x["raw"]) for x in img["sentences"]]

    return train_caps, val_caps, test_caps


def calculate_caption_statistics(captions: list):
    lengths = [len(x.split(" ")) for x in captions]
    return (
        max(lengths),
        min(lengths),
        sum(lengths) / len(lengths),
        lengths.index(max(lengths)),
        lengths.index(min(lengths)),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Caption Counter")
    parser.add_argument(
        "--coco_json",
        required=True,
        type=str,
        help="The location of the Karpathy Split json file",
    )
    args = parser.parse_args()

    assert os.path.exists(args.coco_json), f"Cannot find file {args.coco_json}"

    train_caps, val_caps, test_caps = extract_captions(args.coco_json)

    print("Training Caption Statistics")
    max_cap_len, min_cap_len, avg_cap_len, long_idx, short_idx = (
        calculate_caption_statistics(train_caps)
    )
    print(f"Longest caption: {max_cap_len} -- {train_caps[long_idx]}")
    print(f"Shortest caption: {min_cap_len} -- {train_caps[short_idx]}")
    print(f"Average caption length: {avg_cap_len:.2f}")
    print("\n")

    print("Validation Caption Statistics")
    max_cap_len, min_cap_len, avg_cap_len, long_idx, short_idx = (
        calculate_caption_statistics(val_caps)
    )
    print(f"Longest caption: {max_cap_len} -- {val_caps[long_idx]}")
    print(f"Shortest caption: {min_cap_len} -- {val_caps[short_idx]}")
    print(f"Average caption length: {avg_cap_len:.2f}")
    print("\n")

    print("Test Caption Statistics")
    max_cap_len, min_cap_len, avg_cap_len, long_idx, short_idx = (
        calculate_caption_statistics(test_caps)
    )
    print(f"Longest caption: {max_cap_len} -- {test_caps[long_idx]}")
    print(f"Shortest caption: {min_cap_len} -- {test_caps[short_idx]}")
    print(f"Average caption length: {avg_cap_len:.2f}")
    print("\n")
