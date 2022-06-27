import torch
import torch.nn as nn

from .utils import bbox_jitter, get_bbox_after_aug

__all__ = ['basic']


class BasicNeck(nn.Module):
    def __init__(self, aug_threshold=0., bbox_jitter=None, num_classes=60, multi_class=True):
        super(BasicNeck, self).__init__()
        
        # threshold on preserved ratio of bboxes after cropping augmentation
        self.aug_threshold = aug_threshold
        # config for bbox jittering
        self.bbox_jitter = bbox_jitter

        self.num_classes = num_classes
        self.multi_class = multi_class

    # data: aug_info, labels, filenames, mid_times
    # returns: num_rois, rois, roi_ids, targets, sizes_before_padding, filenames, mid_times, bboxes, bbox_ids
    def forward(self, data):
        roi_ids, targets, sizes_before_padding, filenames, mid_times = [0], [], [], [], []
        bboxes, bbox_ids = [], []  # used for multi-crop fusion
        rois = None
        key_tube_uids = {}

        cur_bbox_id = -1
        roi_id = 0
        for idx in range(len(data['aug_info'])): # idx is batch num
            aug_info = data['aug_info'][idx]
            pad_ratio = aug_info['pad_ratio']
            sizes_before_padding.append([1. / pad_ratio[0], 1. / pad_ratio[1]])
            
            # labels in the key frame
            key_labels = data['batch_labels'][idx][(len(data['batch_labels'][idx]) // 2) - 1]
            for label in key_labels: # set key frame action labels and tube_uids
                cur_bbox_id += 1
                roi_id += 1
                filenames.append(data['filenames'][idx])
                mid_times.append(data['mid_times'][idx])
                bboxes.append(label['bounding_box'])
                bbox_ids.append(cur_bbox_id)

                if self.multi_class:
                    ret = torch.zeros(self.num_classes)
                    ret.put_(torch.LongTensor(label['label']), 
                            torch.ones(len(label['label'])))
                else:
                    ret = torch.LongTensor(label['label'])

                targets.append(ret)

                key_tube_uids[label['tube_uid']] = roi_id - 1
                    
            roi_ids.append(roi_id)
            
            # produce rois according to tube_uids in the key frame
            # len(data['batch_labels'][idx]) is the number of labels in [idx] clip within the batch
            frame_rois = torch.ones(roi_ids[-1] - roi_ids[-2], len(data['batch_labels'][idx]), 5).cuda()
            frame_rois[:, :, 0] = idx
            for frame_idx, frame_labels in enumerate(data['batch_labels'][idx]):
                for label in frame_labels:
                    # jitter is a robustness step, it produces a bbox which varies slightly in size and shape from the original
                    if self.training and self.bbox_jitter is not None:
                        bbox_with_jitter = bbox_jitter(label['bounding_box'],
                                                self.bbox_jitter.get('num', 1),
                                                self.bbox_jitter.scale)
                    else:
                        # no bbox jittering during evaluation
                        bbox_with_jitter = label['bounding_box']
                    
                    bbox = get_bbox_after_aug(aug_info, bbox_with_jitter, self.aug_threshold)
                    if bbox is None:
                        continue
                    frame_rois[key_tube_uids[label['tube_uid']] - roi_ids[-2], frame_idx, 1:] = torch.tensor(bbox)
    
            if rois is None: 
                rois = frame_rois
            else:
                rois = torch.cat((rois, frame_rois), 0)        

        num_rois = len(rois)
        if num_rois == 0:
            return {'num_rois': 0, 'rois': None, 'roi_ids': roi_ids, 'targets': None, 
                    'sizes_before_padding': sizes_before_padding,
                    'filenames': filenames, 'mid_times': mid_times, 'bboxes': bboxes, 'bbox_ids': bbox_ids}
        
        targets = torch.stack(targets, dim=0).cuda()
        return {'num_rois': num_rois, 'rois': rois, 'roi_ids': roi_ids, 'targets': targets, 
                'sizes_before_padding': sizes_before_padding,
                'filenames': filenames, 'mid_times': mid_times, 'bboxes': bboxes, 'bbox_ids': bbox_ids}
    
def basic(**kwargs):
    model = BasicNeck(**kwargs)
    return model
