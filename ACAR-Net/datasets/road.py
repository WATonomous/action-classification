from collections import defaultdict
from PIL import Image
import os
import json
import copy

import torch
import torch.utils.data as data
from .ava import batch_pad, get_aug_info


class ROADDataLoader(data.DataLoader):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 sampler=None,
                 batch_sampler=None,
                 num_workers=0,
                 pin_memory=False,
                 drop_last=False,
                 **kwargs):
        super(ROADDataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs
        )

    def _collate_fn(self, batch):
        clips = [_['clip'] for _ in batch]
        clips, pad_ratios = batch_pad(clips)
        aug_info = []
        for datum, pad_ratio in zip(batch, pad_ratios):
            datum['aug_info']['pad_ratio'] = pad_ratio
            aug_info.append(datum['aug_info'])
        filenames = [_['video_name'] for _ in batch]
        labels = [_['label'] for _ in batch]
        mid_times = [_['mid_time'] for _ in batch]
        bbox_ids = [_['bbox_id_frame'] for _ in batch]
        tube_uids = [_['tube_uid_frame'] for _ in batch]

        output = {
            'clips': clips,
            'aug_info': aug_info,
            'filenames': filenames,
            'labels': labels,
            'mid_times': mid_times,
            'bbox_ids': bbox_ids,
            'tube_uids': tube_uids
        }
        return output


class ROAD(data.Dataset):
    def __init__(self,
                 root_path,
                 annotation_path,
                 class_idx_path,
                 split,
                 save_json=False,
                 spatial_transform=None,
                 temporal_transform=None):
        """
        Dataset class for ROAD data.

        Note that only valid datapoints are included.
        A datapoint (a frame) is valid if it can be a center frame of a clip with
        a number of frames equal to self.num_frames_in_clip.

        We compile the following information for each frame (this is stored in self.data)
        {
            'video': str, name of video,
            'time': int, this index of the frame in the video,
            'clip_start_frame': int, the video centered at the clip_center_frame would start (the id of the ),
            'clip_center_frame': int, the center frame of the clip, and the frame whose agent action classes we are predicting,
            'clip_end_frame': int, where a clip of the video centered at the clip_center_frame would end (the id of the last frame of the clip),
            'format_str': str, format string that will formated into the frame path,
            'frame_rate': int, frames per second,
            'labels': List [
                {
                    'tube_uid': str, tube_uid, e.g. c6a29d05,
                    'bounding_box': list, 4-tuple of floats
                    'label': list, class_idx of the label of this specific box (list of length 1)
                },
                ...
            ]
        }

        root_path : str
            path to the folders of rgb images (each folder being a different video in ROAD)
        annotation_path : str
            path to an annotation file in the same format as the official road dataset annotations
        class_idx_path : str
            path to a json file which contains a mapping from class names to indices
        split : str
            [train_1', 'train_2', 'train_3', 'val_1', 'val_2', or 'val_3']
        spatial_transform : spatial_transforms.Compose, optional
            apply random transformations such as scaling, cropping, flipping, and jittering, by default None
        temporal_transform : temporal_transforms.TemporalCenterRetentionCrop, optional
            See TemporalCenterRetentionCrop, by default None
        """
        self.data = [] # holds all frames in the dataset
        self.valid_tube_indices = []# holds all indices of valid frames
        self.fps = 12
        self.action_counts = defaultdict(int)
        self.total_boxes = 0
        self.num_frames_in_clip = 91
        self.split = split

        with open(class_idx_path, "r") as f:
            items = json.load(f).items()
            self.idx_to_class = sorted(items, key=lambda x: x[1])
            self.idx_to_class = list(
                map(lambda x: {'name': x[0], 'id': x[1]}, self.idx_to_class))

        with open(annotation_path, "r") as f:
            fs = f.read()
            ann_dict = json.loads(fs)
            self.load_data_split(ann_dict)

        self.root_path = root_path
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform

        if save_json:
            self.annotation_path = annotation_path
            self.w_ann_dict = copy.deepcopy(ann_dict)

    def load_data_split(self, ann_dict):
        for video in ann_dict['db'].keys():
            if self.split not in ann_dict['db'][video]['split_ids']:
                continue

            for frame in ann_dict['db'][video]['frames'].values():
                # any frame that contains annotations is a training/val point
                if not frame['annotated'] or len(frame['annos']) == 0:
                    continue

                # Let's use this frame as a training/val point
                # First we compute the information we need to create a clip from this datapoint.
                frame_id = int(frame['input_image_id'])
                # suppose this current frame is treated as the center frame of a clip.
                # note that the frames are expected to be 1-indexed in the annotation file.
                # index of the frame in the video.
                clip_center_frame = frame_id
                clip_start_frame = max(
                    1, frame_id - self.num_frames_in_clip // 2)
                clip_end_frame = min(
                    ann_dict['db'][video]['numf'], frame_id + self.num_frames_in_clip // 2)
                n_frames = clip_end_frame - clip_start_frame + 1

                # then we gather all label information in this frame.
                labels = []
                for bbox_id, annon in frame['annos'].items():
                    for action_id in annon['action_ids']:
                        self.action_counts[action_id] += 1
                    if 'tube_uid' not in annon:
                        # when testing on our own object detector (non ground truth
                        # without an object tracker), we don't have tube_uid, so we infill 
                        # it with the frame id. the effect of this downstream is that no clip frame
                        # other than the midframe will have labels. This is fine as long as
                        # we are using default acar, which only selects the midframe anyways.
                        tube_uid = f"{frame_id}_{action_id}_{self.action_counts[action_id]}"
                    else:
                        tube_uid = annon['tube_uid']
                    labels.append({
                        'tube_uid': tube_uid,
                        'bounding_box': annon['box'],
                        'bbox_id': bbox_id,
                        'label': annon['action_ids']})

                if n_frames == self.num_frames_in_clip:
                    self.valid_tube_indices.append(len(self.data))

                self.data.append({
                    'video': video,
                    'time': frame_id,
                    'clip_start_frame': clip_start_frame,
                    'clip_center_frame': clip_center_frame,
                    'clip_end_frame': clip_end_frame,
                    'format_str': '%05d.jpg',
                    'frame_rate': self.fps,
                    'labels': labels
                })

        # track and print data distribution for potential debugging purposes
        print("Data Distribution by Action Class:")
        for k, v in self.action_counts.items():
            print(self.idx_to_class[k], v)

        print("valid tube indices:", len(self.valid_tube_indices))
        print("total datapoints:", len(self.data))

    def write_to_json(self):
        ''' Dumps annotation dictionary into the desire json file
        '''
        anno_name = os.path.splitext(self.annotation_path)[0]
        new_annotation_path = os.path.join(self.annotation_path, '..', anno_name + '_acar.json')

        if hasattr(self, 'w_ann_dict'):
            with open(new_annotation_path, "w") as json_writer: # writer for the new anno dict file 
                json.dump(self.w_ann_dict, json_writer)
        else:
            raise RuntimeError('write_to_json called but no writer was present')
    
    def write_actions(self, outputs, video_name, frame_time, bbox_ids, num_rois):
        ''' Appends action labels onto their respective bbox in the given frame
        '''
        action_classes = list(range(outputs.shape[1]))
        for k in range(num_rois):
            self.w_ann_dict['db'][video_name[k]]['frames'][frame_time[k]]['annos'][bbox_ids[k]]['action_scores']  = {}

            for cls in action_classes:
                score = float(outputs[k][cls])
                self.w_ann_dict['db'][video_name[k]]['frames'][frame_time[k]]['annos'][bbox_ids[k]]['action_scores'][cls + 1] = score
                

    def _spatial_transform(self, clip):
        if self.spatial_transform is not None:
            init_size = clip[0].size[:2]
            params = self.spatial_transform.randomize_parameters()
            aug_info = get_aug_info(init_size, params)

            clip = [self.spatial_transform(img) for img in clip]
        else:
            aug_info = None
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        return clip, aug_info

    def __getitem__(self, index):
        """Return data during training

        Parameters
        ----------
        index : int
            index of the example in the data.

        Returns
        -------
        dict
            clip: List[torch.Tensor], Tensor.shape is (3,32,256,341), rgb_images of frames
            aug_info: List[Dict], info about the augmentations done to the clips
            label: List[List[Dict]] for each frame in clip (length 32), contains a list of annotations
                in the frame, overall representing the labels of all frames in the tube.
            video_name: str, name of the video
            mid_time: str, frame_id of the key frame
            bbox_id_frame: bbox ids of the key frame
            tube_uid_frame: tube uids of the key frame

        Raises
        ------
        RuntimeError
            there may be be errors that occur when trying to load actual images into memory.
        """
        index = self.valid_tube_indices[index]
        path = os.path.join(self.root_path, self.data[index]['video'])
        frame_format = self.data[index]['format_str']
        clip_start_frame = self.data[index]['clip_start_frame']
        clip_center_frame = self.data[index]['clip_center_frame']
        clip_end_frame = self.data[index]['clip_end_frame']
        mid_time = str(self.data[index]['time'])
        video_name = self.data[index]['video']

        clip_labels = []
        keyframe_tube_uids = [label['tube_uid'] for label in self.data[index]['labels']]

        # frame indices are offsets from 'index'
        frame_indices = list(range(clip_start_frame - clip_center_frame, clip_end_frame - clip_center_frame))

        if self.temporal_transform is not None:
            # the center frame offset is 0
            # this temporal transform must be TemporalCenterRetentionCrop
            frame_indices = self.temporal_transform(frame_indices, 0)
            assert frame_indices.index(0) == (len(frame_indices) // 2) - 1

        # get bbox_ids of the keyframe
        keyframe_bbox_ids = [label['bbox_id'] for label in self.data[index]['labels']]

        clip = []
        for i in range(len(frame_indices)):

            clip_labels.append([label for label in self.data[index + frame_indices[i]]['labels']
                                if label['tube_uid'] in keyframe_tube_uids])

            image_path = os.path.join(path, frame_format % (clip_center_frame + frame_indices[i]))
            try:
                with Image.open(image_path) as img:
                    img = img.convert('RGB')
            except BaseException as e:
                raise RuntimeError(
                    'Caught "{}" when loading {}'.format(str(e), image_path))
            clip.append(img)

        clip, aug_info = self._spatial_transform(clip)

        return {'clip': clip, 'aug_info': aug_info, 'label': clip_labels,
                'video_name': video_name, 'mid_time': mid_time, 'bbox_id_frame': keyframe_bbox_ids,
                'tube_uid_frame': keyframe_tube_uids}

    def __len__(self):
        return len(self.valid_tube_indices)

class ROADmulticropDataLoader(ROADDataLoader):
    def _collate_fn(self, batch):
        clips, aug_info = [], []
        for i in range(len(batch[0]['clip'])):
            clip, pad_ratios = batch_pad([_['clip'][i] for _ in batch])
            clips.append(clip)
            cur_aug_info = []
            for datum, pad_ratio in zip(batch, pad_ratios):
                datum['aug_info'][i]['pad_ratio'] = pad_ratio
                cur_aug_info.append(datum['aug_info'][i])
            aug_info.append(cur_aug_info)
        filenames = [_['video_name'] for _ in batch]
        labels = [_['label'] for _ in batch]
        mid_times = [_['mid_time'] for _ in batch]
        bbox_ids = [_['bbox_id_frame'] for _ in batch]
        tube_uids = [_['tube_uid_frame'] for _ in batch]

        output = {
            'clips': clips,
            'aug_info': aug_info,
            'filenames': filenames,
            'labels': labels,
            'mid_times': mid_times,
            'bbox_ids': bbox_ids,
            'tube_uids': tube_uids
        }
        return output


############################################################
# function logic below is identical to AVA 
############################################################

class ROADmulticrop(ROAD):
    def _spatial_transform(self, clip):
        if self.spatial_transform is not None:
            assert isinstance(self.spatial_transform, list)

            init_size = clip[0].size[:2]
            clips, aug_info = [], []
            for st in self.spatial_transform:
                params = st.randomize_parameters()
                aug_info.append(get_aug_info(init_size, params))

                clips.append(torch.stack([st(img)
                             for img in clip], 0).permute(1, 0, 2, 3))
        else:
            aug_info = [None]
            clips = [torch.stack(clip, 0).permute(1, 0, 2, 3)]
        return clips, aug_info
