import argparse
import yaml
import numpy as np
from tqdm import tqdm

from easydict import EasyDict
from road_video.road_video import ROADDebugVideo

def main(args):
    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    opts = EasyDict(config)

    video_builder = ROADDebugVideo(opts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCSORT Tube Generation and Eval')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    main(args)