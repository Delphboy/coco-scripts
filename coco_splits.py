import json
import argparse
import os

train = []
val = []
test = []


def load_file(file_name: str) -> dict:
    with open(file_name, "r") as f:
        data = json.load(f)
    return data


def main(file_name, output_location, percentage):
    data = load_file(file_name)
    images = data["images"]
    for img in images:
        if img["split"] == "val":
            val.append(img)
        elif img["split"] == "test":
            test.append(img)
        else:
            train.append(img)

    mini_train = train[: int(len(train) * percentage)]
    # mini_val = val[: int(len(val) * PERCENTAGE)]
    # mini_test = test[: int(len(test) * PERCENTAGE)]

    new_data = {}
    new_data["images"] = mini_train + val + test
    new_data["dataset"] = data["dataset"]

    print(f"Number of image in train: {len(mini_train)}")
    print(f"Number of image in val: {len(val)}")
    print(f"Number of image in test: {len(test)}")
    print(f"Number of image in new dataset split: {len(new_data["images"])}")

    # Output file
    out_file = os.path.join(
        output_location, f"new_dataset_coco_{int(percentage * 100)}.json"
    )
    with open(out_file, "w+") as f:
        json.dump(new_data, f)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--dataset_json",
        type=str,
        help="Location of the dataset json file",
        required=True,
    )
    args.add_argument(
        "--output_location",
        type=str,
        help="The directory to dump the file",
        required=True,
    )
    args.add_argument(
        "--percentage",
        type=float,
        help="Percentage of full dataset in the new split (between 0 and 1)",
        default=0.1,
    )
    args = args.parse_args()

    main(args.dataset_json, args.output_location, args.percentage)
