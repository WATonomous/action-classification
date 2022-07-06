import json
import os

import torch.utils.data as data

class ROADOCSORT(data.Dataset):
    def __init__(self, annotation_path, save_tubes, ground_truth):
        ''' ROAD Dataset class:
        Reads ROAD annotations and provides the minimum compatible with OC_SORT. Saves produced tubes into new json annotation file
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

        for frame in self.ann_dict['db'][video]['frames'].values():
            dp = {}
            dp['frame_id'] = frame['rgb_image_id']
            dp['frame_labels'] = []

            if frame['annotated']:
                for annon in frame['annos'].values():
                    # 1 is the confidence score only if annon box is the ground truth,
                    # ocsort is actually built to handle multiple detections of varying confidence
                    conf = lambda x: 1 if x else annon['conf']

                    dp['frame_labels'].append({
                        'box': annon['box'].append(conf(ground_truth)), 
                        'tube_uid': annon['tube_uid'], 
                        'agent_ids': annon['agent_ids']
                        }) 

                    if annon['agent_ids'][0] not in self.data_dict[video]['agent_types']:
                        self.data_dict[video]['agent_types'].append(annon['agent_ids'])

            self.data_dict[video]['frames'].append(dp)

    def __setitem__(self, index, frames, agent_types):
        ''' Sets video with new labels and tube tracks for each frame. 
            
            Args:
                index: index of video (same as __getitem__)
                frames: same length as video, contains the detections and their tube uids
                    - structure: [dp, ...]
        '''
        video_key = self.data_dict.keys()[index]
        self.data_dict[video_key]['frames'] = frames
        self.data_dict[video_key]['agent_types'] = agent_types

        # if we are saving tubes, edit the ann_dict
        if hasattr(self, 'new_annotation_path'):  
            ''' ROADTube takes in:
                - {'tube_uid': annon['tube_uid'], 'bounding_box': annon['box'], 'label': annon['action_ids']}
                where annon is in ann_dict['db'][video_key][frame_key]['annos']
                - frame['rgb_image_id']
                where frame is in ann_dict['db'][video_key]['frames']
            ''' 
            for dp in frames:
                self.ann_dict['db'][video_key][dp['frame_id']]['annos'].clear()

                if len(dp['frame_labels']) is not 0:
                    self.ann_dict['db'][video_key][dp['frame_id']]['annotated'] = True
                    # convert list of annotations to a dictionary of them, save into ann_dict
                    for anno in dp['frame_labels']:
                        bbox_name = f'b{self.bbox_counter:06}'
                        # by replacing annos, we are removing loc_ids, action_ids, duplex_ids, triplet_ids
                        # these are unused annotations 
                        self.ann_dict['db'][video_key][dp['frame_id']]['annos'][bbox_name] = anno
                        self.bbox_counter += 1

                else:
                    self.ann_dict['db'][video_key][dp['frame_id']]['annotated'] = False

    def __getitem__(self, index): 
        ''' Returns:
                video_name: name of the video
                frame_detections: list of frames and their detections
        '''
        # video_key = self.data_dict.keys()[index]
        # return {'video_name': video_key, 'video_data': self.data_dict[video_key]}
        ''' returns a video with a list of frames. each frame has a list of bounding boxes and their agent and confidence score.
        '''

    def __len__(self):
        return len(self.data_dict.keys())