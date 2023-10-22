#!/bin/sh
# source ~/.bashrc
# conda activate pgp

python nuscenes_dataset_match.py --cfg_file data_extraction/nuscenes_dataset_occ.yaml --type car &
python nuscenes_dataset_match.py --cfg_file data_extraction/nuscenes_dataset_occ.yaml --skip_compute_stats --type bus &
python nuscenes_dataset_match.py --cfg_file data_extraction/nuscenes_dataset_occ.yaml --skip_compute_stats --type truck &
python nuscenes_dataset_match.py --cfg_file data_extraction/nuscenes_dataset_occ.yaml --skip_compute_stats --type trailer