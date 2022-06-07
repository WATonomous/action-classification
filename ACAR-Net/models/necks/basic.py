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
    # TODO change the number of rois coming into here, make sure that each group of rois share the same id
    # invalid roi is [batch_num, 1, 1, 1, 1]
    def forward(self, data):
        rois, roi_ids, targets, sizes_before_padding, filenames, mid_times = [], [0], [], [], [], []
        bboxes, bbox_ids = [], []  # used for multi-crop fusion

        cur_bbox_id = -1  # record current bbox no.
        for idx in range(len(data['aug_info'])): # idx is batch num
            aug_info = data['aug_info'][idx]
            pad_ratio = aug_info['pad_ratio']
            sizes_before_padding.append([1. / pad_ratio[0], 1. / pad_ratio[1]])
            
            # TODO add another loop here I think
            ''' BASED ON WHAT YOU HAVE DONE TO THE DATA LOADER, MAKE THE NECESSARY CHANGES TO ITERATE THROUGH EACH FRAME
            '''
            # for label in data['labels'][idx]: 
            #     ''' label:
            #         [{tube_uid, bounding_box, label (action_label)}, ...]
            #     '''
            #     cur_bbox_id += 1
            #     if self.training and self.bbox_jitter is not None:
            #         bbox_list = bbox_jitter(label['bounding_box'],
            #                                 self.bbox_jitter.get('num', 1),
            #                                 self.bbox_jitter.scale)
            #     else:
            #         # no bbox jittering during evaluation
            #         bbox_list = [label['bounding_box']] # label needs to have multiple bboxes for each frame (ditto to the other part of the conditional statement)
                
            #     for b in bbox_list:
            #         bbox = get_bbox_after_aug(aug_info, b, self.aug_threshold)
            #         if bbox is None:
            #             continue
            #         rois.append([idx] + bbox) # this needs to change roi.append([idx] + bbox_frame_list)
                    
            #         filenames.append(data['filenames'][idx])
            #         mid_times.append(data['mid_times'][idx])
            #         bboxes.append(label['bounding_box'])
            #         bbox_ids.append(cur_bbox_id)

            #         if self.multi_class:
            #             ret = torch.zeros(self.num_classes)
            #             ret.put_(torch.LongTensor(label['label']), 
            #                     torch.ones(len(label['label'])))
            #         else:
            #             ret = torch.LongTensor(label['label'])
            #         targets.append(ret)
                
            # roi_ids.append(len(rois))
            key_frame_labels = data['batch_labels'][idx][len(data['batch_labels'][idx]) // 2]

            for label in key_frame_labels: # labels in the key frame
                ''' label:
                    [{tube_uid, bounding_box, label (action_label)}, ...]
                '''
                cur_bbox_id += 1
                if self.training and self.bbox_jitter is not None:
                    bbox_list = bbox_jitter(label['bounding_box'],
                                            self.bbox_jitter.get('num', 1),
                                            self.bbox_jitter.scale)
                else:
                    # no bbox jittering during evaluation
                    bbox_list = [label['bounding_box']] # label needs to have multiple bboxes for each frame (ditto to the other part of the conditional statement)
                
                for b in bbox_list:
                    bbox = get_bbox_after_aug(aug_info, b, self.aug_threshold)
                    if bbox is None:
                        continue
                    rois.append([idx] + bbox) # this needs to change roi.append([idx] + bbox_frame_list)
                    
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
                
            roi_ids.append(len(rois))


        num_rois = len(rois)
        if num_rois == 0:
            return {'num_rois': 0, 'rois': None, 'roi_ids': roi_ids, 'targets': None, 
                    'sizes_before_padding': sizes_before_padding,
                    'filenames': filenames, 'mid_times': mid_times, 'bboxes': bboxes, 'bbox_ids': bbox_ids}
        
        rois = torch.FloatTensor(rois).cuda()
        targets = torch.stack(targets, dim=0).cuda()
        return {'num_rois': num_rois, 'rois': rois, 'roi_ids': roi_ids, 'targets': targets, 
                'sizes_before_padding': sizes_before_padding,
                'filenames': filenames, 'mid_times': mid_times, 'bboxes': bboxes, 'bbox_ids': bbox_ids}
    
def basic(**kwargs):
    model = BasicNeck(**kwargs)
    return model
