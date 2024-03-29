import os
import json
import numpy as np
import copy
import motmetrics as mm

from .eval_utils import UIDtoNumber, xyxy2xywh
mm.lap.default_solver = 'lap'

LABEL_LENGTH = 5 # dp['frame_labels'] is [[x1, y1, x2, y2, tube_uid], ...]

class ROADMOTEvaluator(object):
    ''' ROAD MOT EVALUATOR: loads in detections and tube_uids from the ROAD dataset, 
        evaluates on the online targets generated by OCSORT.

        Args:
            annotation_path: path to the ROAD dataset annotations
            accumulate: determines whether the evaluator should accumulate the score
                throughout all the videos. Otherwise, each video gets a score.
    '''
    def __init__(self, annotation_path):
        self.eval_dict = {}
        ''' Eval Dictionary: {
                video_name: {
                    'frames': [[x1, y1, x2, y2, tube_uid], ...],  
                    'num_frames': int(number of frames in the video)
                }
            }
        '''

        with open(annotation_path, "r") as f:
            fs = f.read()
            self.ann_dict = json.loads(fs) # annotation dictionary

            for video in self.ann_dict['db'].keys():
                self.uid2number = UIDtoNumber()
                self.append_new_data(video)     

        self.reset_accumulator()

    def append_new_data(self, video):
        ''' takes in the video name and annotation dictionary to create a list of 
            consecutive frames for this video in the eval_dict, each frame is 
            a list of detections and their tube_uid.
        '''
        self.eval_dict[video] = {} # initialize
        self.eval_dict[video]['frames'] = []
        self.eval_dict[video]['num_frames'] = self.ann_dict['db'][video]['numf']

        for frame in self.ann_dict['db'][video]['frames'].values():
            dp = {}
            dp['frame_id'] = frame['rgb_image_id']
            dp['frame_labels'] = np.empty((0, LABEL_LENGTH))

            if frame['annotated']:
                for annon in frame['annos'].values():
                    # dp['frame_labels'] is [[x1, y1, x2, y2, tube_uid], ...]
                    xywh = xyxy2xywh(np.array(annon['box'], dtype=object))
                    tube_id = self.uid2number.uid2number(annon['tube_uid'])
                    xywh = np.append(xywh, tube_id)
                    dp['frame_labels'] = np.append(dp['frame_labels'], [xywh], axis=0)

            self.eval_dict[video]['frames'].append(dp)

    def reset_accumulator(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame(self, trk_xywhs, trk_ids, gt_xywhs, gt_ids, rtn_events=False):
        # get distance matrix
        iou_distance = mm.distances.iou_matrix(gt_xywhs, trk_xywhs, max_iou=0.5)

        # acc
        self.acc.update(gt_ids, trk_ids, iou_distance)

        if rtn_events and iou_distance.size > 0 and hasattr(self.acc, 'last_mot_events'):
            events = self.acc.last_mot_events  # only supported by https://github.com/longcw/py-motmetrics
        else:
            events = None
        return events

    def eval_video(self, gt_key, video):
        """ video coming in is 
                np.array([
                        array([[x1, y1, x2, y2, tube_uid, agent_id], ...]), 
                        array([[x1, y1, x2, y2, tube_uid, agent_id], ...]), 
                        ...
                        ])
        """
        self.reset_accumulator()

        gt_video = self.eval_dict[gt_key] # ground truth annotations

        for i, frame in enumerate(video):
            frame[:, :4] = xyxy2xywh(frame[:, :4]) # [[x, y, w, h, tube_id, agent_id], ...]

            trk_xywhs = frame[:, :4]
            trk_ids = frame[:, 4]

            gt_xywhs = gt_video['frames'][i]['frame_labels'][:, :4]
            gt_ids = gt_video['frames'][i]['frame_labels'][:, 4]

            self.eval_frame(trk_xywhs, trk_ids, gt_xywhs, gt_ids, rtn_events=False)

        return self.acc

    @staticmethod
    def get_summary(accs, names, metrics=('mota', 'num_switches', 'idp', 'idr', 'idf1', 'precision', 'recall')):
        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)

        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs,
            metrics=metrics,
            names=names,
            generate_overall=True
        )

        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )

        return strsummary

    @staticmethod
    def save_summary(summary, filename):
        import pandas as pd
        writer = pd.ExcelWriter(filename)
        summary.to_excel(writer)
        writer.save()
