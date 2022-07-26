from __future__ import annotations
import argparse
import yaml
import numpy as np
from tqdm import tqdm

from easydict import EasyDict
from datasets.road_dataset import ROADOCSORT
from trackers.ocsort_tracker.ocsort import OCSort
from trackers.tracking_utils.evaluation import ROADMOTEvaluator

ACTION_MATCHING_THRESH = 1e-5

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
        annotation_path=opt.Data.annotation_path, 
        save_tubes=opt.save_tubes,
        ground_truth=opt.Data.ground_truth
    )

    if opt.Evaluation.evaluate:
        evaluator = ROADMOTEvaluator(
            annotation_path=opt.Evaluation.evaluation_path, 
            accumulate=opt.Evaluation.accumulate,
        )

        accumulators = []

    # enumerate through data, produce tubes
    for idx, data in enumerate(dataloader):
        ''' data contains {'video_name': string, 'video_data': {'frames': [frame_annos], 'agent_types': [list of agent types]}}
            where frame_annos is {'frame_id': int, 'frame_labels': [[x1, y1, x2, y2, score, agent, action_ids], ...]}
        '''

        # create separate trackers for each of the different agents, this is to avoid
        # ID switches across agents
        agent_trackers = [OCSort(opt.OCSORT) for _ in data['video_data']['agent_types']]

        # run the trackers all at once on each of the frames (passing agents that they are concerned with)
        online_targets = []
        target_actions = [] # this keeps track of the action ids of the boxes [[action_ids per target], ...]

        # progress through the video
        if opt.progress:
            print(f"{data['video_name']}:")
            progress = tqdm(total=len(data['video_data']['frames']), ncols=100)

        for frame in data['video_data']['frames']:
            # tqdm is slower than ocsort
            if opt.progress:
                progress.update()

            # dict of masks for each agent
            frame_masks = {agent_id: (frame['frame_labels'][:, 5] == agent_id) for agent_id in data['video_data']['agent_types']}

            # send each tracker their respective detections, consolidate tracks and their agent id
            # online_target: np.array([[x1, y1, x2, y2, tube_uid, agent_id, action_ids], ...]) which are the tracked detections in a frame
                # online_targets is num_frame # of online_target ^^
            online_target = None

            if opt.Data.match_actions: target_action = []

            for agent_id, agent_tracker in zip(data['video_data']['agent_types'], agent_trackers):
                agent_frame_labels = frame['frame_labels'][frame_masks[agent_id]] # masks out the boxes that aren't the agent we want

                output_results = np.array(agent_frame_labels[:, :5], dtype=float)
                if output_results.size == 0:
                    output_results = np.empty((0, 5))

                targets = agent_tracker.update(output_results)
                
                # post processing to combine targets with their agent again
                agent_ids = np.reshape(np.full(len(targets), agent_id), (len(targets), 1))
                agent_targets = np.concatenate((targets, agent_ids), axis=1)
                if online_target is None:
                    online_target = agent_targets
                else:
                    online_target = np.concatenate((online_target, agent_targets), axis=0) 

            if opt.Data.match_actions:# matching targets with their original action ids
                for target in online_target:
                    target_index = 0
                    for i_frame in frame['frame_labels']:
                        if (np.abs(i_frame[:4] - target[:4]) < float(opt.Data.action_matching_thresh)).all():
                            break
                        
                        target_index += 1

                    target_action.append(frame['frame_labels'][:, 6][target_index])
            
            if opt.Data.match_actions: target_actions.append(target_action)

            online_targets.append(online_target)

        if opt.progress:
            progress.close()

        # if debug, then save a 
        if opt.debug_frames:
            pass

        # if save, pass tracks back into dataloader to save into ann_dict
        if opt.save_tubes:
            dataloader[idx] = (online_targets, target_actions)

        # if evaluate, evaluate on the ground truth tubes provided by ROAD
        if opt.Evaluation.evaluate:
            # if no error occurs here, than python weird and doesn't care that a library 
            # is not imported
            accumulators.append(evaluator.eval_video(idx, online_targets[:, :4]))

    if opt.Evaluation.evaluate: # summarize evaluation, perhaps write it too if needed
        for accumulator in accumulators:
            print(evaluator.get_summary(accumulator, 'accumulator'))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCSORT Tube Generation and Eval')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    main(args)