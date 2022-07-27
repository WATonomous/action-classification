import torch
import torch.nn as nn

from .utils import bbox_jitter, get_bbox_after_aug

__all__ = ['tube']


class TubeNeck(nn.Module):
    def __init__(self, aug_threshold=0., bbox_jitter=None, num_classes=60, multi_class=True):
        super(TubeNeck, self).__init__()
        
        # threshold on preserved ratio of bboxes after cropping augmentation
        self.aug_threshold = aug_threshold
        # config for bbox jittering
        self.bbox_jitter = bbox_jitter

        self.num_classes = num_classes
        self.multi_class = multi_class

    # data: aug_info, labels, filenames, mid_times
    # returns: num_rois, rois, roi_ids, targets, sizes_before_padding, filenames, mid_times, bboxes, bbox_ids
    def forward(self, data):
        """Gathers rois for all the keyframes (elements in the batch) and their target
        labels.

        Parameters
        ----------
        data : dict
            {'aug_info': parameters used for data augmentation, 
             'labels': labels for the whole tube,
             'filenames': filenames for each example, 
             'mid_times': mid_times for each example}

        Returns
        -------

        dict
            {'num_rois': total number of rois, 
             'rois': all the rois we are concerned with 
                    (each label has 32 bboxes corrosponding to each fast frame), 
                    frames with no information about the keyframe labels also have corresponding 
                    rois, filled with ones. Shape will be (num_rois, 32, 5)
                    ex. [a, a, a, b, b, c, c, c, c]
                    Where all a belong to the same batch element, all b belong to the same
                    batch element, etc. 
             'roi_ids': where each batch of rois end in the list of rois, 
                    ex. In the above, roi_ids would be [0, 3, 5, 9].
                    This used to easily index the rois tensor later in the acar head.
             'targets': the target labels for the keyframes with shape (batch_size, num_classes),
             'sizes_before_padding': image size before padding, used later in the ACAR-head.
             'filenames': filenames for each example, 
             'mid_times': mid_times for each example, 
             'bboxes': bounding boxes unique within the batch, 
             'bbox_ids': bounding box ids unique within the batch}
        """
        roi_ids, targets, sizes_before_padding, filenames, mid_times = [0], [], [], [], []
        bboxes, bbox_ids = [], []  # used for multi-crop fusion
        rois = None
        # this map stores the order in which we see the labels in the keyframe
        # so that it can be used as the index at which corresponding tube rois
        # are inserted in the variable frame_rois.
        key_tube_uids = {}

        cur_bbox_id = -1
        roi_id = 0
        for idx in range(len(data['aug_info'])): # idx is batch num
            aug_info = data['aug_info'][idx]
            pad_ratio = aug_info['pad_ratio']
            sizes_before_padding.append([1. / pad_ratio[0], 1. / pad_ratio[1]])
            
            # labels in the key frame
            key_labels = data['labels'][idx][(len(data['labels'][idx]) // 2) - 1]
            for label in key_labels: # set key frame action labels and tube_uids
                cur_bbox_id += 1
                roi_id += 1
                filenames.append(data['filenames'][idx])
                mid_times.append(data['mid_times'][idx])
                bboxes.append(label['bounding_box'])
                bbox_ids.append(cur_bbox_id)
    
                if self.multi_class:
                    # constructing the target tensor
                    ret = torch.zeros(self.num_classes)
                    ret.put_(torch.LongTensor(label['label']), 
                            torch.ones(len(label['label'])))
                else:
                    ret = torch.LongTensor(label['label'])

                targets.append(ret)

                key_tube_uids[label['tube_uid']] = roi_id - 1
                    
            roi_ids.append(roi_id)
            
            # produce rois according to tube_uids in the key frame
            # len(data['labels'][idx]) is the number of labels in [idx] clip within the batch
            # first create a tensor of shape (num_key_labels, 32, 5)
            frame_rois = torch.ones(roi_ids[-1] - roi_ids[-2], len(data['labels'][idx]), 5)
            # the first number in the 5-tuple (the last dimension) is the index of the keyframe in the batch
            # this is used to track the rois down the line
            frame_rois[:, :, 0] = idx
            for frame_idx, frame_labels in enumerate(data['labels'][idx]):
                for label in frame_labels:
                    # As a design decision part of the tube improvements to the ACAR-Net model, 
                    # we will enforce this. We already have lots of rois due to the tubes, 
                    # if we were to also jitter each roi, there would be more complex considerations
                    # down the line. For now, we will try to keep this simple. 
                    assert self.bbox_jitter.num == 1
                    # jitter is a robustness step, it produces a bbox which varies slightly in size and shape from the original
                    if self.training and self.bbox_jitter is not None:
                        bbox_with_jitter = bbox_jitter(label['bounding_box'],
                                                self.bbox_jitter.get('num', 1),
                                                self.bbox_jitter.scale)[0]
                    else:
                        # no bbox jittering during evaluation
                        bbox_with_jitter = label['bounding_box']
                    
                    bbox = get_bbox_after_aug(aug_info, bbox_with_jitter, self.aug_threshold)
                    if bbox is None:
                        continue
                    # put the frame roi in the right place in the tensor
                    # the second last element in roi_id is the the index at which the
                    # rois for the keyframes at this batch element begins
                    frame_rois[key_tube_uids[label['tube_uid']] - roi_ids[-2], frame_idx, 1:] = torch.tensor(bbox)
    
            # we concatenate frame_rois for each batch element, so that rois will eventually
            # be an ordered tensor of all rois within the batch.
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
        return {'num_rois': num_rois, 'rois': rois.cuda(), 'roi_ids': roi_ids, 'targets': targets, 
                'sizes_before_padding': sizes_before_padding,
                'filenames': filenames, 'mid_times': mid_times, 'bboxes': bboxes, 'bbox_ids': bbox_ids}
    
def tube(**kwargs):
    model = TubeNeck(**kwargs)
    return model
