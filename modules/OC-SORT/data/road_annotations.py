import json
import os
import numpy as np

import torch.utils.data as data

LABEL_LENGTH = 7 # dp['frame_labels'] is [[x1, y1, x2, y2, score, agent, action_ids], ...]

class ROADOCSORT(data.Dataset):
    def __init__(self, opts, save_tubes):
        ''' ROAD Dataset class:
        Reads ROAD annotations and provides the minimum compatible with OC_SORT and the
        evaluation of it. Saves produced tubes into new json annotation file
        if instructed to.  

        Args:
            annotation_path: path to the annotations
            ground_truth: is the annotation path the ground truth?
            ground_truth_path: path to the ground truth
            class_idx_path: path to the class indexes
            save_tubes: instructs the class to initialize a copy of the annotations file and save new tube_uids to it
        
        Annotation Dictionary:
            ROAD format, can either have detections of confidence 1
        '''
        self.data_dict = {} # data dictionary for videos and each of their frames, this is the minimum we need to pass
        self.bbox_counter = 0 # used when saving tubes, counts total bboxes saved
        self.ground_truth = opts.ground_truth
        self.match_actions = opts.match_actions

        with open(opts.annotation_path, "r") as f:
            fs = f.read()
            self.ann_dict = json.loads(fs) # annotation dictionary

            for video in self.ann_dict['db'].keys():
                self.append_new_data(video)     

        if not self.ground_truth:
            with open(opts.ground_truth_path, "r") as f:
                fs = f.read()
                self.gt_dict = json.loads(fs) # ground_truth dictionary

        if save_tubes:
            # get the name of the annotation file 
            anno_name = os.path.splitext(opts.annotation_path)[0] 
            self.new_annotation_path = os.path.join(opts.annotation_path, '..', anno_name + '_ocsort.json')

            # writer for the new anno dict file 
            self.w = open(self.new_annotation_path, "w")
    
    def __del__(self):
        ''' If a new_annotation_path exists, then we dump the contents of ann_dict into it
        '''
        if hasattr(self, 'new_annotation_path'):
            json.dump(self.ann_dict, self.w)    

    def append_new_data(self, video): 
        ''' takes in the video name and annotation dictionary to create a list of 
            consecutive frames for this video in the data_dict, each frame is 
            a list of detections and their scores.
        '''
        self.data_dict[video] = {} # initialize
        self.data_dict[video]['frames'] = []
        self.data_dict[video]['agent_types'] = []
        self.data_dict[video]['num_frames'] = self.ann_dict['db'][video]['numf']

        for frame in self.ann_dict['db'][video]['frames'].values():
            dp = {}
            if self.ground_truth:
                dp['frame_id'] = frame['rgb_image_id']
            else:
                dp['frame_id'] = frame['input_image_id']

            dp['frame_labels'] = np.empty((0, LABEL_LENGTH))

            if len(frame['annos']) > 0:
                for annon in frame['annos'].values():
                    # 1 is the confidence score only if annon box is the ground truth,
                    # ocsort is actually built to handle multiple detections of varying confidence
                    conf = lambda x: 1 if x else annon['score']
                    
                    # dp['frame_labels'] is [[x1, y1, x2, y2, score, agent, action_ids], ...]
                    box = list(annon['box'])
                    box.append(conf(self.ground_truth))
                    box.append(annon['agent_ids'][0])
                    box.append(annon['action_ids'])

                    dp['frame_labels'] = np.append(dp['frame_labels'], [box], axis=0)

                    if annon['agent_ids'][0] not in self.data_dict[video]['agent_types']:
                        self.data_dict[video]['agent_types'].append(annon['agent_ids'][0])

            self.data_dict[video]['frames'].append(dp)

    def write_tracks(self, video_key, input):
        ''' Sets video with new labels and tube tracks for each frame. 
            
            Args:
                name: name of video 
                input: a tuple of online_targets and target_actions
        '''
        online_targets, target_actions = input

        # if we are saving tubes, edit the ann_dict
        if hasattr(self, 'new_annotation_path'):  
            ''' ROADTube takes in:
                - {'tube_uid': annon['tube_uid'], 'bounding_box': annon['box'], 'label': annon['action_ids']}
                where annon is in ann_dict['db'][video_key][frame_key]['annos']
                - frame['rgb_image_id']
                where frame is in ann_dict['db'][video_key]['frames']
            ''' 
            for idx, online_target in enumerate(online_targets):
                # Deletes the annos key-value pair to free up space for the track detections
                try:
                    self.ann_dict['db'][video_key]['frames'][str(idx+1)]['annotated'] = \
                        self.gt_dict['db'][video_key]['frames'][str(idx+1)]['annotated']
                except KeyError: # this is for when we create tubes for the test set
                    self.ann_dict['db'][video_key]['frames'][str(idx+1)]['annotated'] = 1

                if not self.ann_dict['db'][video_key]['frames'][str(idx+1)]['annotated']:
                    continue
                
                if 'annos' in self.ann_dict['db'][video_key]['frames'][str(idx + 1)]:
                    del self.ann_dict['db'][video_key]['frames'][str(idx + 1)]['annos']

                # convert list of annotations to a dictionary of them, save into ann_dict
                self.ann_dict['db'][video_key]['frames'][str(idx + 1)]['annos'] = {}
                
                if self.match_actions:
                    target_action = target_actions[idx]
                else:
                    target_action = [[1] for _ in online_target] # set to 1 as a fake ID for ACAR's calc_mAP, this design should change...

                for target, action in zip(online_target, target_action):
                    self.bbox_counter += 1
                    bbox_name = f"b{self.bbox_counter:06}"
                    # by replacing annos, we are removing loc_ids, action_ids, duplex_ids, triplet_ids
                    # these are unused annotations 
                    anno = {}
                    anno['box'] = list(target[:4])
                    anno['tube_uid'] = target[4]
                    anno['agent_ids'] = target[5]
                    anno['action_ids'] = action
                    self.ann_dict['db'][video_key]['frames'][str(idx + 1)]['annos'][bbox_name] = anno

    def get_video_names(self):
        return list(self.ann_dict['db'].keys())

    def __getitem__(self, index): 
        ''' Returns:
                video_name: name of the video
                frame_detections: list of frames and their detections
        '''
        video_key = list(self.data_dict.keys())[index]
        return {'video_name': video_key, 'video_data': self.data_dict[video_key], 
                'num_frames': self.data_dict[video_key]['num_frames']}


    def __len__(self):
        return len(self.data_dict.keys())