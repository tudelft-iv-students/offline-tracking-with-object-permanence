from pathlib import Path
import os

from data_extraction.data_extraction import extract_data
from nuscenes.eval.common.config import config_factory

os.environ["MKL_NUM_THREADS"] = "18"
os.environ["NUMEXPR_NUM_THREADS"] = "18"
os.environ["OMP_NUM_THREADS"] = "18"

if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='extract_data', help='')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='')
    parser.add_argument('--result_path', type=str, default='mot_results/v1.0-test/tracking_result.json', help='')
    parser.add_argument('--tracker_name', type=str, default='CenterPoint', help='tracker name')
    parser.add_argument('--data_root', type=str, required=True, help='nuscenes dataroot')
    args = parser.parse_args()

    if args.func == 'extract_data':
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        dataset_cfg.VERSION = args.version
        cfg_ = config_factory('tracking_nips_2019')
        cfg_.class_names=['car',
                        'bus',
                        'truck',
                        'trailer']
        
        extract_data(
            version=args.version,
            data_path=args.data_root,
            save_path=(Path(__file__).resolve().parent ).resolve() / 'extracted_mot_data'/'final_version_nms'/args.tracker_name,
            max_sweeps=dataset_cfg.MAX_SWEEPS,
            result_path=args.result_path,
            cfg=cfg_
        )

