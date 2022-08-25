from easydict import EasyDict
import yaml
import json
import argparse

import data.transforms as vtf

from torchvision import transforms
from evaluation import format_acar_dets, eval_framewise_dets, build_eval_tubes
from data import VideoDataset
from utils import utils

def main(args):
    """ POST-PROCESSING AND EVALUATION
        Takes in the json annotations coming out of ACAR and processes them into 
        data compatible with 3D-RetinaNet's ROAD-Evaluation code
    """
    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    opt = EasyDict(config)

    args = utils.set_args(args) # set directories and SUBSETS for datasets

    skip_step = args.SEQ_LEN*8

    val_transform = transforms.Compose([ 
                        vtf.ResizeClip(args.MIN_SIZE, args.MAX_SIZE),
                        vtf.ToTensorStack(),
                        vtf.Normalize(mean=args.MEANS,std=args.STDS)])

    val_dataset = VideoDataset(args, train=False, transform=val_transform, skip_step=skip_step, full_test=full_test)

    format_acar_dets(opt.formatting)
    eval_framewise_dets(opt.eval.framewise)
    build_eval_tubes(opt.eval.videowise)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Post-processing and Eval')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    main(args)