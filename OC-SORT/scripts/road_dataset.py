import json
import os
import numpy as np

import torch.utils.data as data

LABEL_LENGTH = 6 # dp['frame_labels'] is [[x1, y1, x2, y2, score, agent], ...]

class ROADOCSORT(data.Dataset):
    def __init__(self, annotation_path, save_tubes, ground_truth, evaluate):
        ''' ROAD Dataset class:
        Reads ROAD annotations and provides the minimum compatible with OC_SORT and the
        evaluation of it. Saves produced tubes into new json annotation file
        if instructed to.  

        Args:
            root_path: path to the folders of rgb images (each folder being a different video in ROAD)
            annotation_path: path to the annotations
            class_idx_path: path to the class indexes
            save_tubes: instructs the class to initialize a copy of the annotations file and save new tube_uids to it
        
        Annotation Dictionary:
            ROAD format, can either have detections of confidence 1
        '''
        self.data_dict = {} # data dictionary for videos and each of their frames, this is the minimum we need to pass
        self.eval_dict = {}
        self.bbox_counter = 0 # used when saving tubes, counts total bboxes saved

        with open(annotation_path, "r") as f:
            fs = f.read()
            self.ann_dict = json.loads(fs) # annotation dictionary

            for video in self.ann_dict['db'].keys():
                self.append_new_data(video, ground_truth)     

        if save_tubes:
            # get the name of the annotation file 
            anno_name = os.path.splitext(annotation_path)[0] 
            self.new_annotation_path = os.path.join(annotation_path, '..', anno_name + 'ocsort.json')

            # writer for the new anno dict file 
            self.w = open(self.new_annotation_path, "w")

        if evaluate:
            pass
    
    def __del__(self):
        ''' If a new_annotation_path exists, then we dump the contents of ann_dict into it
        '''
        if hasattr(self, 'new_annotation_path'):
            json.dump(self.ann_dict, self.w)    

    def append_new_data(self, video, ground_truth): 
        ''' takes in the video name and annotation dictionary to create a list of 
            consecutive frames for this video in the data_dict, each frame is 
            a list of detections and their scores.
        '''
        self.data_dict[video] = {} # initialize
        self.data_dict[video]['frames'] = []
        self.data_dict[video]['agent_types'] = []
        self.data_dict[video]['num_frames'] = self.ann_dict['db']['video']['numf']

        for frame in self.ann_dict['db'][video]['frames'].values():
            dp = {}
            dp['frame_id'] = frame['rgb_image_id']
            dp['frame_labels'] = np.empty((0, LABEL_LENGTH))

            if frame['annotated']:
                for annon in frame['annos'].values():
                    # 1 is the confidence score only if annon box is the ground truth,
                    # ocsort is actually built to handle multiple detections of varying confidence
                    conf = lambda x: 1 if x else annon['conf']
                    
                    # dp['frame_labels'] is [[x1, y1, x2, y2, score, agent], ...]
                    dp['frame_labels'] = np.append(dp['frame_labels'], 
                        annon['box'].append(conf(ground_truth)).append(annon['agent_ids'])
                        ) 

                    if annon['agent_ids'][0] not in self.data_dict[video]['agent_types']:
                        self.data_dict[video]['agent_types'].append(annon['agent_ids'])

                dp['frame_labels'] = np.array(dp['frame_labels'])

            self.data_dict[video]['frames'].append(dp)

    def __setitem__(self, index, online_targets):
        ''' Sets video with new labels and tube tracks for each frame. 
            
            Args:
                index: index of video (same as __getitem__)
                frames: same length as video, contains the detections and their tube uids
                    - structure: [[x1, y1, x2, y2, tube_uid, agent_id], ...]
        '''
        # if we are saving tubes, edit the ann_dict
        if hasattr(self, 'new_annotation_path'):  
            ''' ROADTube takes in:
                - {'tube_uid': annon['tube_uid'], 'bounding_box': annon['box'], 'label': annon['action_ids']}
                where annon is in ann_dict['db'][video_key][frame_key]['annos']
                - frame['rgb_image_id']
                where frame is in ann_dict['db'][video_key]['frames']
            ''' 
            video_key = self.data_dict.keys()[index]

            for idx, online_target in enumerate(online_targets):
                self.ann_dict['db'][video_key][idx]['annos'].clear()

                if len(online_target) is not 0:
                    self.ann_dict['db'][video_key][idx]['annotated'] = True
                    # convert list of annotations to a dictionary of them, save into ann_dict
                    for target in online_target:
                        bbox_name = f'b{self.bbox_counter:06}'
                        # by replacing annos, we are removing loc_ids, action_ids, duplex_ids, triplet_ids
                        # these are unused annotations 
                        anno = {}
                        anno['bounding_box'] = target[:3]
                        anno['tube_uid'] = target[4]
                        anno['agent_ids'] = target[5]
                        self.ann_dict['db'][video_key][idx]['annos'][bbox_name] = anno
                        self.bbox_counter += 1

                else:
                    self.ann_dict['db'][video_key][dp['frame_id']]['annotated'] = False

    def __getitem__(self, index): 
        ''' Returns:
                video_name: name of the video
                frame_detections: list of frames and their detections
        '''
        video_key = self.data_dict.keys()[index]
        return {'video_name': video_key, 'video_data': self.data_dict[video_key], 
                'num_frames': self.data_dict[video_key]['num_frames']}
        ''' returns a video with a list of frames. each frame has a list of bounding boxes and their agent and confidence score.
        '''


    def __len__(self):
        return len(self.data_dict.keys())