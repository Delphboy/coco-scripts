# COCO Scripts
A set of scripts for the COCO dataset


## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install numpy
```

## Bottom Up Top Down

The BUTD features can be downloaded following the instructions [here](https://github.com/peteanderson80/bottom-up-attention). 

 - `make_bu_data.py`: Extracts the features out of the `tsv` files and creates the required directories. See `make_bu_data.qsub` for an example of how the script is called. 
 - `prepro_labels.py`

## Shell

- `coco-download.sh`: Downloads the COCO dataset images and the Karpathy Split JSON file