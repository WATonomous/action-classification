from collections import defaultdict
from PIL import Image
import os
import json

import torch
import torch.utils.data as data
from .ava import batch_pad, get_aug_info

class ROADDataLoader(data.DataLoader):
    def __init__(self,
                 dataset,
                 tube_labels,
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
        self.tube_labels = tube_labels

    def _collate_fn(self, batch):
        clips = [_['clip'] for _ in batch]
        clips, pad_ratios = batch_pad(clips)
        aug_info = []
        for datum, pad_ratio in zip(batch, pad_ratios):
            datum['aug_info']['pad_ratio'] = pad_ratio
            aug_info.append(datum['aug_info'])
        filenames = [_['video_name'] for _ in batch]
        if self.tube_labels:
            # these are labels for each tube
            labels = [_['clip_labels'] for _ in batch]
        else:
            # these are the labels only for each frame.
            labels = [_['label'] for _ in batch] 
        mid_times = [_['mid_time'] for _ in batch]
        
        output = {
            'clips': clips,
            'aug_info': aug_info,
            'filenames': filenames,
            'labels': labels,
            'mid_times': mid_times
        }
        return output


class ROAD(data.Dataset):

    def __init__(self,
                 root_path,
                 annotation_path,
                 class_idx_path,
                 split,
                 spatial_transform=None,
                 temporal_transform=None):
        self.data = []
        self.fps = 12
        self.action_counts = defaultdict(int)
        self.total_boxes = 0
        self.num_frames_in_clip = 91

        with open(annotation_path, "r") as f:
            
            fs = f.read()
            ann_dict = json.loads(fs)

            if split == "train_1":
                for video in ann_dict['db'].keys():
                    if split not in ann_dict['db'][video]['split_ids']:
                        continue

                    for frame in ann_dict['db'][video]['frames'].values():
                        if not frame['annotated'] or len(frame['annos']) == 0: # any frame that contains annotations is a training point
                            continue
                        # Let's use this frame as a training point
                        dp = {}
                        frame_id = int(frame['input_image_id'])
                        if frame_id % 1 != 0:
                            continue
                        dp['video'] = video
                        dp['time'] = frame_id
                        dp['midframe'] = frame_id
                        dp['start_frame'] = max(0, frame_id - self.num_frames_in_clip // 2)
                        dp['n_frames'] = self.num_frames_in_clip
                        if dp['start_frame'] + dp['n_frames'] - 1 > ann_dict['db'][video]['numf']:
                            dp['n_frames'] = ann_dict['db'][video]['numf'] - dp['start_frame'] + 1
                        dp['format_str'] = '%05d.jpg'
                        dp['frame_rate'] = self.fps
                        dp['labels'] = []
                        assert len(frame['annos']) > 0, frame['annotated']
                        for annon in frame['annos'].values():
                            self.total_boxes += 1
                            label = {'bounding_box': annon['box'], 'label': annon['action_ids']}
                            for action_id in annon['action_ids']:
                                self.action_counts[action_id] += 1
                            dp['labels'].append(label)
                        self.data.append(dp)
                print("Data Distribution by action class:")
                for k, v in self.action_counts.items():
                    print(ann_dict['all_action_labels'][k], v)
            elif split == "val_1":
                for video in ann_dict['db'].keys():
                    if split not in ann_dict['db'][video]['split_ids']:
                        continue
                    for frame in ann_dict['db'][video]['frames'].values():
                        if not frame['annotated'] or len(frame['annos']) == 0:
                            continue
                        # Let's use this frame as a validation point
                        dp = {}
                        frame_id = int(frame['input_image_id'])
                        dp['video'] = video
                        dp['time'] = frame_id
                        dp['midframe'] = frame_id
                        dp['start_frame'] = max(0, frame_id - self.num_frames_in_clip // 2)
                        dp['n_frames'] = self.num_frames_in_clip
                        if dp['start_frame'] + dp['n_frames'] - 1 > ann_dict['db'][video]['numf']:
                            dp['n_frames'] = ann_dict['db'][video]['numf'] - dp['start_frame'] + 1
                        dp['format_str'] = '%05d.jpg'
                        dp['frame_rate'] = self.fps
                        dp['labels'] = [] 
                        assert len(frame['annos']) > 0, frame['annotated']
                        for annon in frame['annos'].values():
                            label = {'bounding_box': annon['box'], 'label': annon['action_ids']}
                            for action_id in annon['action_ids']:
                                self.action_counts[action_id] += 1
                            dp['labels'].append(label)
                        self.data.append(dp)

        with open(class_idx_path, "r") as f:
            items = json.load(f).items()
            self.idx_to_class = sorted(items, key=lambda x: x[1])
            self.idx_to_class = list(map(lambda x: {'name': x[0], 'id': x[1]}, self.idx_to_class))

        self.root_path = root_path
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform

    def detection_bbox_to_ava(self, bbox):
        x1, y1, x2, y2 = bbox
        convbb = [x1/1280, y1/960, x2/1280, y2/960]
        return convbb
        

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
        path = os.path.join(self.root_path, self.data[index]['video'])
        frame_format = self.data[index]['format_str']
        start_frame = self.data[index]['start_frame']
        n_frames = self.data[index]['n_frames']
        mid_time = str(self.data[index]['time'])
        target = self.data[index]['labels']
        video_name = self.data[index]['video']
        
        frame_indices = list(range(start_frame, start_frame + n_frames))
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        
        clip = []
        for i in range(len(frame_indices)):
            image_path = os.path.join(path, frame_format%frame_indices[i])
            try:
                with Image.open(image_path) as img:
                    img = img.convert('RGB')
            except BaseException as e:
                raise RuntimeError('Caught "{}" when loading {}'.format(str(e), image_path))
            clip.append(img)

        clip, aug_info = self._spatial_transform(clip)
        
        return {'clip': clip, 'aug_info': aug_info, 'label': target, 
                'video_name': video_name, 'mid_time': mid_time}

    def __len__(self):
        return len(self.data)

class ROADTube(data.Dataset):
    def __init__(self,
                 root_path,
                 annotation_path,
                 class_idx_path,
                 stride,
                 split,
                 spatial_transform=None,
                 temporal_transform=None):
        """ROADTube Dataset class:
        Splits the ROAD dataset into clips of video. Processes annotations and loads frame images
        in a format that is compatible with the model.

        Parameters
        ----------
        root_path : str
            path to the folders of rgb images (each folder being a different video in ROAD)
        annotation_path : str
            path to the offical road dataset annotations
        class_idx_path : str
            path to a mapping from class names to indices
        stride : int
            downsampling, stride means more uniqueness in data but less data points
        split : str
            train_1', 'train_2', 'train_3', 'val_1', 'val_2', or 'val_3'
        spatial_transform : spatial_transforms.Compose, optional
            apply random transformations such as scaling, cropping, flipping, and jittering, by default None
        temporal_transform : temporal_transforms.TemporalCenterRetentionCrop, optional
            See TemporalCenterRetentionCrop, by default None
        """

        self.data = [] # stores the other frames
        self.data_stride = [] # stores the stride'th frame
        self.fps = 12
        self.num_frames_in_clip = 91
        self.stride = stride 

        with open(annotation_path, "r") as f:
            fs = f.read()
            ann_dict = json.loads(fs) 
            for video in ann_dict['db'].keys():
                if split not in ann_dict['db'][video]['split_ids']:
                    continue

                self.append_new_data(video, ann_dict)            

        with open(class_idx_path, "r") as f:
            items = json.load(f).items()
            self.idx_to_class = sorted(items, key=lambda x: x[1])
            self.idx_to_class = list(map(lambda x: {'name': x[0], 'id': x[1]}, self.idx_to_class))

        self.root_path = root_path
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform

    def append_new_data(self, video, ann_dict):
        """Takes in the video name and annotation dictionary to create data points, 
        appending them to self.data and self.data_stride

        Parameters
        ----------
        video : str
            name of video to add new data from.
        ann_dict : dict
            dictionary generated from loading the official ROAD dataset annotations.
        """
        stride_counter = 0
        for frame in ann_dict['db'][video]['frames'].values():
            # each dp is a datapoint, i.e. one training example
            dp = {}
            frame_id = int(frame['rgb_image_id'])
            dp['video'] = video
            dp['time'] = frame_id
            dp['n_frames'] = self.num_frames_in_clip

            dp['format_str'] = '%05d.jpg'
            dp['frame_rate'] = self.fps
            dp['frame_labels'] = []
            
            if frame['annotated']:
                labeled = False
                for annon in frame['annos'].values():
                    labeled = True
                    label = {'tube_uid': annon['tube_uid'], 'bounding_box': annon['box'], 'label': annon['action_ids']}
                    dp['frame_labels'].append(label)
                
                stride_counter += 1

                if stride_counter % self.stride == 0 and labeled: # every stride'th labeled frame is a datapoint
                    stride_counter = 0

                    # if clip made from stride'th frame is incomplete, we do not consider it 
                    # when a labeled frame exists near the beginning or end of the data,
                    # the clip produced using that specific keyframe is incomplete
                    if frame_id + (self.num_frames_in_clip // 2) < ann_dict['db'][video]['numf'] \
                            and frame_id - (self.num_frames_in_clip // 2) >= 0:

                        dp['center_frame'] = len(self.data)
                        self.data_stride.append(dp)

            self.data.append(dp)

    def _spatial_transform(self, clip):
        # identical to the same function in the normal ROAD dataset
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
            _description_

        Returns
        -------
        dict
            clip: rgb_images of frames
            aug_info: some sort of info about the augmentations done to the clips
            clip_labels: labels in each of the frames that pertain to the tube_uid of the center_frame
            video_name: name of the video
            mid_time: frame_id of the key frame

        Raises
        ------
        RuntimeError
            there may be be errors that occur when trying to load actual images into memory.
        """

        path = os.path.join(self.root_path, self.data_stride[index]['video'])
        frame_format = self.data_stride[index]['format_str']
        center_frame = self.data_stride[index]['center_frame']
        n_frames = self.data_stride[index]['n_frames']
        mid_time = str(self.data_stride[index]['time'])
        key_tube_uids = [label['tube_uid'] # must remain constant going into ACAR Neck
            for label in self.data_stride[index]['frame_labels']]
        clip_labels = []
        video_name = self.data_stride[index]['video']
        
        start_frame = center_frame - n_frames // 2
        end_frame = center_frame + n_frames // 2

        frame_indices = list(range(start_frame, end_frame))
        # propagate center index into temporal transform in order to conserve label
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices, center_frame)
        
        clip = []
        # load frames in a clip, consolidate agent tracks via tube_uid
        for i in range(len(frame_indices)):
            # append the labels in a frame if that label is part of an agent track in the key frame
            clip_labels.append([label for label in self.data[frame_indices[i]]['frame_labels'] 
                if label['tube_uid'] in key_tube_uids])

            image_path = os.path.join(path, frame_format%self.data[frame_indices[i]]['time'])
            try:
                with Image.open(image_path) as img:
                    img = img.convert('RGB')
            except BaseException as e:
                raise RuntimeError('Caught "{}" when loading {}'.format(str(e), image_path))
            clip.append(img)

        clip, aug_info = self._spatial_transform(clip)
        
        return {'clip': clip, 'aug_info': aug_info, 'clip_labels': clip_labels, 
                'video_name': video_name, 'mid_time': mid_time}

    def __len__(self):
        return len(self.data_stride)


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
        if self.tube_labels:
            # these are labels for each tube
            labels = [_['clip_labels'] for _ in batch]
        else:
            # these are the labels only for each frame.
            labels = [_['label'] for _ in batch] 
        mid_times = [_['mid_time'] for _ in batch]
        
        output = {
            'clips': clips,
            'aug_info': aug_info,
            'filenames': filenames,
            'labels': labels,
            'mid_times': mid_times
        }
        return output


############################################################
# function logic below is identical to AVA and to each other
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
            
                clips.append(torch.stack([st(img) for img in clip], 0).permute(1, 0, 2, 3))
        else:
            aug_info = [None]
            clips = [torch.stack(clip, 0).permute(1, 0, 2, 3)]
        return clips, aug_info

class ROADTubemulticrop(ROADTube):
    def _spatial_transform(self, clip):
        if self.spatial_transform is not None:
            assert isinstance(self.spatial_transform, list)
                      
            init_size = clip[0].size[:2]
            clips, aug_info = [], []
            for st in self.spatial_transform:
                params = st.randomize_parameters()
                aug_info.append(get_aug_info(init_size, params))
            
                clips.append(torch.stack([st(img) for img in clip], 0).permute(1, 0, 2, 3))
        else:
            aug_info = [None]
            clips = [torch.stack(clip, 0).permute(1, 0, 2, 3)]
        return clips, aug_info