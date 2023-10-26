# Offline Tracking with Object Permanence

## Introduction

This repository contains code for ["Offline Tracking with Object Permanence "](https://arxiv.org/abs/2310.01288) by Xianzhong Liu, Holger Caesar.  This project aims to recover the occluded vehicle trajectories and reduce the identity switches caused by occlusions.

```bibtex
@misc{liu2023offline,
      title={Offline Tracking with Object Permanence}, 
      author={Xianzhong Liu and Holger Caesar},
      year={2023},
      eprint={2310.01288},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Visualiztion

<img src="assets/vis1.gif" width="450" height="450"/>
<img src="assets/vis4.gif" width="450" height="450"/>

* Rectangels: GT boxes.

* <span style="color:blue">Blue arrows</span>: recovered box centers which are originally missing in the initial tracking result. 

* <span style="color:red">Red arrows</span>: visible box centers in initial online tracking.






## Installation

1. Clone this repository 

2. Set up a new conda environment 
``` shell
conda create --name offline_trk python=3.7
```

3. Install dependencies
```shell
conda activate offline_trk

# nuScenes devkit
pip install nuscenes-devkit

# Pytorch: The code has been tested with Pytorch 1.7.1, CUDA 10.1, but should work with newer versions
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch

# Additional utilities
pip install psutil
pip install positional-encodings
pip install imageio
pip install tensorboard
```


## Dataset

1. Download the [nuScenes dataset](https://www.nuscenes.org/download). For this project we just need the following.
    - Metadata for the Trainval split (v1.0)
    - Map expansion pack (v1.3)

2. Organize the nuScenes root directory as follows
```plain
└── nuScenes/
    ├── maps/
    |   ├── basemaps/
    |   ├── expansion/
    |   ├── prediction/
    |   ├── 36092f0b03a857c6a3403e25b4b7aab3.png
    |   ├── 37819e65e09e5547b8a3ceaefba56bb2.png
    |   ├── 53992ee3023e5494b90c316c183be829.png
    |   └── 93406b464a165eaba6d9de76ca09f5da.png
    └── v1.0-trainval
        ├── attribute.json
        ├── calibrated_sensor.json
        ...
        └── visibility.json         
```

3. Run the following script to extract pre-processed data. This speeds up training significantly.
```shell
python preprocess.py -c configs/preprocess_nuscenes.yml -r path/to/nuScenes/root/directory -d path/to/directory/with/preprocessed/data
```


## Inference with online tracking result


### Generating the initial online tracking result
1. Download the [detection results](https://mitprod-my.sharepoint.com/:f:/g/personal/tianweiy_mit_edu/Eip_tOTYSk5JhdVtVzlXlyABDPnGx9vsnwdo5SRK7bsh8w?e=vSdija) in standard nuScenes submission format. (Note: the link is from [CenterPoint](https://github.com/tianweiy/CenterPoint). Any other detectors will also work as long as it fits the format.) The detection results can be saved in [det_result](./det_results/).
2. Run the tracking script (TODO: add multiprocessing to make the nms faster)
```shell
python nusc_tracking/pub_test.py --work_dir mot_results  --checkpoint det_results/your_detection_result(json file) --nms --version v1.0-test
```
### Extract vehicle tracklets and convert to input format for Re-ID
3. Extract vehicle tracklets
```
python initial_extraction.py --cfg_file data_extraction/nuscenes_dataset_occ.yaml --version v1.0-test  --result_path mot_results/v1.0-test/tracking_result.json
``` 
4. Convert to Re-ID input(TODO: add multiprocessing to make the extraction faster)
```
##Slower
#python nuscenes_dataset_match.py --cfg_file data_extraction/nuscenes_dataset_occ.yaml
##OR a more faster way
bash Re-ID_extraction.sh
```
### Performing Re-ID
5. Reassociate history tracklets with future tracklets by changing the tracking ID of the future tracklets
```
python motion_matching.py --cfg_file motion_associator/re-association.yaml --result_path mot_results/v1.0-test/tracking_result.json
```

6. To visualize all the association result, run
```
python motion_matching.py --cfg_file motion_associator/re-association.yaml --result_path mot_results/v1.0-test/tracking_result.json --visualize
```
The plots will be stored under `./mot_results/Re-ID_results/matching_info/v1.0-test/plots`. Note that some times the detections are flipped for 180 degrees. 
<p align="center">
    <img src="assets/association_vis.png" width="450" height="450"/>
</p>

* <span style="color:green">Green arrows</span>: History tracklet. 

* <span style="color:blue">Blue arrows</span>: Future tracklets with low association scores. 


* <span style="color:red">Red arrows</span>: Future tracklets with high association scores. 

## Training

To train the Re-ID model from scratch, run
```shell
python train.py -c configs/match_train_augment.yml -r path/to/nuScenes/root/directory -d path/to/directory/with/preprocessed/Re-ID/data -o path/to/output/directory/Re-ID -n 50
```

To train the track completion model from scratch, run
```shell
python train.py -c configs/track_completion.yml -r path/to/nuScenes/root/directory -d path/to/directory/with/preprocessed/Track_completion/data -o path/to/output/directory/Track_completion -n 50
```

The training script will save training checkpoints and tensorboard logs in the output directory.

To launch tensorboard, run
```shell
tensorboard --logdir=path/to/output/directory/tensorboard_logs
```
