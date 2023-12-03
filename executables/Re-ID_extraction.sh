path=$1
tracker=$2

python executables/nuscenes_dataset_match.py --cfg_file data_extraction/nuscenes_dataset_occ.yaml --tracker_name ${tracker} --type car --data_root ${path} &
python executables/nuscenes_dataset_match.py --cfg_file data_extraction/nuscenes_dataset_occ.yaml --tracker_name ${tracker} --type bus --data_root ${path} &
python executables/nuscenes_dataset_match.py --cfg_file data_extraction/nuscenes_dataset_occ.yaml --tracker_name ${tracker} --type truck --data_root ${path} &
python executables/nuscenes_dataset_match.py --cfg_file data_extraction/nuscenes_dataset_occ.yaml --tracker_name ${tracker} --type trailer --data_root ${path} 