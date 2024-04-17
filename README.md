# COCO Scripts
A set of scripts for the COCO dataset. Any `.qsub` files are examples of HPC scripts that can be used to download required files and run necessary scripts. 


## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install numpy
```

## Bottom Up Top Down (`butd/`)

The BUTD features can be downloaded following the instructions [here](https://github.com/peteanderson80/bottom-up-attention). 

 - `make_bu_data.py`: Extracts the features out of the `tsv` files and creates the required directories. See `make_bu_data.qsub` for an example of how the script is called. 
 - `prepro_labels.py`: Used by some codebases to preprocess the captions

## Visual Semantic Unit Alignment/Scene Graph Autoencoders (`vsua/`)

The python scripts required to 


## Shell (`shell/`)

- `coco-download.sh`: Downloads the COCO dataset images and the Karpathy Split JSON file

## Original sources:

- [`butd/make_bu_data.py`](https://github.com/ruotianluo/self-critical.pytorch/blob/master/scripts/make_bu_data.py)
- [`butd/prepro_labels.py`](https://github.com/ruotianluo/self-critical.pytorch/blob/master/scripts/prepro_labels.py)
- [`vsua/build_geometry_graph.py`](https://github.com/ltguo19/VSUA-Captioning/blob/master/scripts/build_geometry_graph.py)
- [`vsua/cal_geometry_feats.py`](https://github.com/ltguo19/VSUA-Captioning/blob/master/scripts/cal_geometry_feats.py)