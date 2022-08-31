from easydict import EasyDict
import yaml
import argparse

import data.transforms as vtf

from torchvision import transforms
from evaluation.format_acar_dets import format_acar_dets
from evaluation.gen_dets import eval_framewise_dets
from evaluation.tubes import build_eval_tubes
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

    opt = utils.set_args(opt) # set directories and SUBSETS for datasets

    skip_step = opt.SEQ_LEN*8

    val_transform = transforms.Compose([ 
                        vtf.ResizeClip(opt.MIN_SIZE, opt.MAX_SIZE),
                        vtf.ToTensorStack(),
                        vtf.Normalize(mean=opt.MEANS,std=opt.STDS)])

    val_dataset = VideoDataset(opt, train=False, transform=val_transform, skip_step=skip_step, full_test=True)

    opt.label_types = val_dataset.label_types
    opt.num_label_types = val_dataset.num_label_types
    opt.all_classes =  val_dataset.all_classes
    opt.num_classes_list = val_dataset.num_classes_list
    opt.num_ego_classes = val_dataset.num_ego_classes
    opt.ego_classes = val_dataset.ego_classes

    format_acar_dets(opt.formatting)
    eval_framewise_dets(opt, val_dataset)
    build_eval_tubes(opt, val_dataset)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Post-processing and Eval')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    main(args)