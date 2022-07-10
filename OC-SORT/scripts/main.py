import argparse
import yaml
import numpy as np

from easydict import EasyDict
from road_dataset import ROADOCSORT
from trackers.ocsort_tracker.ocsort import OCSort
from trackers.tracking_utils.evaluation import Evaluator

LABEL_LENGTH = 6 # dp['frame_labels'] is [[x1, y1, x2, y2, score, agent], ...]

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
        ground_truth=opt.ground_truth,
        evaluate=opt.evaluate
    )

    # enumerate through data, produce tubes
    for idx, data in enumerate(dataloader):
        ''' data contains {'video_name': string, 'video_data': {'frames': [frame_annos], 'agent_types': [list of agent types]}}
            where frame_annos is {'box', 'tube_uid', 'agent_ids'}
        '''

        # create separate trackers for each of the different agents, this is to avoid
        # ID switches across agents
        agent_trackers = [OCSort(opt.track_thresh, iou_threshold=opt.iou_thresh, delta_t=opt.deltat, 
            asso_func=opt.asso, inertia=opt.inertia) for _ in data['video_data']['agent_types']]

        # run the trackers all at once on each of the frames (passing agents that they are concerned with)
        online_targets = []
        for frame in data['video_data']['frames'].values():
            # list of masks for each agent
            frame_masks = [(frame[:, 4] == agent_id) for agent_id in data['video_data']['agent_types']]
            # send each tracker their respective detections, consolidate tracks and their agent id
            online_targets.append(np.append([np.append(agent_tracker.update(frame[frame_masks[i]]), i) 
                    for i, agent_tracker in zip(data['video_data']['agent_types'], agent_trackers)]))
        
        # online_target: np.array([[x1, y1, x2, y2, tube_uid, agent_id], ...]) which are the tracked detections in a frame
        # online_targets is num_frame # of online_target ^^

        # if save pass tracks back into dataloader to save into ann_dict
        if opt.save_tubes:
            dataloader[idx] = online_targets

        # if evaluate, evaluate on the ground truth tubes provided by ROAD
        if opt.evaluate:
            pass
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCSORT Tube Generation and Eval')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    main(args)