path=$1

python nuscenes_dataset_match.py --cfg_file data_extraction/nuscenes_dataset_occ.yaml --type car --data_root ${path} &
python nuscenes_dataset_match.py --cfg_file data_extraction/nuscenes_dataset_occ.yaml --type bus --data_root ${path} &
python nuscenes_dataset_match.py --cfg_file data_extraction/nuscenes_dataset_occ.yaml --type truck --data_root ${path} &
python nuscenes_dataset_match.py --cfg_file data_extraction/nuscenes_dataset_occ.yaml --type trailer --data_root ${path}