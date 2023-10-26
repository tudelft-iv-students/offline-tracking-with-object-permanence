#!/bin/sh
# source ~/.bashrc
# conda activate pgp

python nuscenes_dataset_match.py --cfg_file data_extraction/nuscenes_dataset_occ.yaml --type car &
python nuscenes_dataset_match.py --cfg_file data_extraction/nuscenes_dataset_occ.yaml --type bus &
python nuscenes_dataset_match.py --cfg_file data_extraction/nuscenes_dataset_occ.yaml --type truck &
python nuscenes_dataset_match.py --cfg_file data_extraction/nuscenes_dataset_occ.yaml --type trailer