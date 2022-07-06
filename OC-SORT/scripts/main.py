import argparse
import yaml
import numpy as np

from easydict import EasyDict
from road_dataset import ROADOCSORT
from trackers.ocsort_tracker.ocsort import OCSort
from trackers.tracking_utils.evaluation import Evaluator


def main(args):
    """ TUBE GENERATOR
        Takes in detections in ROAD format, and produces another json file containing tube uids.
        The output json file is compatible with the ROADTube dataloader in ACAR-Net
        TODO: potentially look at siamese comparisons.. depends on how good this implementation is
    """
    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    opt = EasyDict(config)

    # load data
    dataloader = ROADOCSORT(
        root_path=opt.root_path, 
        annotation_path=opt.annotation_path, 
        save_tubes=opt.save_tubes,
        ground_truth=opt.ground_truth
    )

    # enumerate through data, produce tubes
    for idx, data in enumerate(dataloader):
        ''' data contains {'video_name': string, 'video_data': {'frames': [frame_annos], 'agent_types': [list of agent types]}}
            where frame_annos is {'box', 'tube_uid', 'agent_ids'}
        '''

        # create separate trackers for each of the different agents, this is to avoid
        # ID switches across agents
        agent_trackers = [OCSort() for _ in data['video_data']['agent_types']]

        # run the trackers all at once on each of the frames (passing agents that they are concerned with)
        for frame in data['video_data']['frames'].values():
            # grab the boxes from the frame, 

            pass
        # grab tracks from all the trackers
        # consolidate

        # if save pass tracks back into dataloader to save into ann_dict

        # if evaluate, evaluate on the ground truth tubes provided by ROAD

        pass
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCSORT Tube Generation and Eval')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    main(args)