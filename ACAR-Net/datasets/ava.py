from distutils.log import ERROR
from PIL import Image
import os
import pickle
import json
from cv2 import sepFilter2D
import numpy as np
import io
from iopath.common.file_io import g_pathmgr

import torch
import torch.nn.functional as F
import torch.utils.data as data


# bounding box order: [left, top, right, bottom]
# size order: [width, height]
def get_aug_info(init_size, params):
    size = init_size
    bbox = [0.0, 0.0, 1.0, 1.0]
    flip = False
    
    for t in params:
        if t is None:
            continue
            
        if t['transform'] == 'RandomHorizontalFlip':
            if t['flip']:
                flip = not flip
            continue
        
        if t['transform'] == 'Scale':
            if isinstance(t['size'], int):
                w, h = size
                if (w <= h and w == t['size']) or (h <= w and h == t['size']):
                    continue
                if w < h:
                    ow = t['size']
                    oh = int(t['size'] * h / w)
                    size = [ow, oh]
                else:
                    oh = t['size']
                    ow = int(t['size'] * w / h)
                    size = [ow, oh]
            else:
                size = t['size']
            continue
            
        if t['transform'] == 'CenterCrop':
            w, h = size
            size = t['size']
            
            x1 = int(round((w - size[0]) / 2.))
            y1 = int(round((h - size[1]) / 2.))
            x2 = x1 + size[0]
            y2 = y1 + size[1]
            
        elif t['transform'] == 'CornerCrop':
            w, h = size
            size = [t['size']] * 2

            if t['crop_position'] == 'c':
                th, tw = (t['size'], t['size'])
                x1 = int(round((w - tw) / 2.))
                y1 = int(round((h - th) / 2.))
                x2 = x1 + tw
                y2 = y1 + th
            elif t['crop_position'] == 'tl':
                x1 = 0
                y1 = 0
                x2 = t['size']
                y2 = t['size']
            elif t['crop_position'] == 'tr':
                x1 = w - self.size
                y1 = 0
                x2 = w
                y2 = t['size']
            elif t['crop_position'] == 'bl':
                x1 = 0
                y1 = h - t['size']
                x2 = t['size']
                y2 = h
            elif t['crop_position'] == 'br':
                x1 = w - t['size']
                y1 = h - t['size']
                x2 = w
                y2 = h
            
        elif t['transform'] == 'ScaleJitteringRandomCrop':
            min_length = min(size[0], size[1])
            jitter_rate = float(t['scale']) / min_length
            
            w = int(jitter_rate * size[0])
            h = int(jitter_rate * size[1])
            size = [t['size']] * 2
            
            x1 = t['pos_x'] * (w - t['size'])
            y1 = t['pos_y'] * (h - t['size'])
            x2 = x1 + t['size']
            y2 = y1 + t['size']
            
        dl = float(x1) / w * (bbox[2] - bbox[0])
        dt = float(y1) / h * (bbox[3] - bbox[1])
        dr = float(x2) / w * (bbox[2] - bbox[0])
        db = float(y2) / h * (bbox[3] - bbox[1])
        
        if flip:
            bbox = [bbox[2] - dr, bbox[1] + dt, bbox[2] - dl, bbox[1] + db]
        else:
            bbox = [bbox[0] + dl, bbox[1] + dt, bbox[0] + dr, bbox[1] + db]

    return {'init_size': init_size, 'crop_box': bbox, 'flip': flip}


def batch_pad(images, alignment=1, pad_value=0):
    max_img_h = max([_.size(-2) for _ in images])
    max_img_w = max([_.size(-1) for _ in images])
    target_h = int(np.ceil(max_img_h / alignment) * alignment)
    target_w = int(np.ceil(max_img_w / alignment) * alignment)
    padded_images, pad_ratios = [], []
    for image in images:
        src_h, src_w = image.size()[-2:]
        pad_size = (0, target_w - src_w, 0, target_h - src_h)
        padded_images.append(F.pad(image, pad_size, 'constant', pad_value).data)
        pad_ratios.append([target_w / src_w, target_h / src_h])
    return torch.stack(padded_images), pad_ratios


class AVADataLoader(data.DataLoader):
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
        super(AVADataLoader, self).__init__(
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
        labels = [_['label'] for _ in batch] # <----------------------- where labels are loaded... batch? oh yeah its list comprehension, but double check what batch is
        ''' you can move onto the main.py after this
        '''
        mid_times = [_['mid_time'] for _ in batch]
        
        output = {
            'clips': clips,
            'aug_info': aug_info,
            'filenames': filenames,
            'labels': labels,
            'mid_times': mid_times
        }
        return output

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
        batch_labels = [_['clip_labels'] for _ in batch] 
        mid_times = [_['mid_time'] for _ in batch]
        
        output = {
            'clips': clips,
            'aug_info': aug_info,
            'filenames': filenames,
            'batch_labels': batch_labels,
            'mid_times': mid_times
        }
        return output

    
class AVA(data.Dataset):
    def __init__(self,
                 root_path,
                 annotation_path,
                 spatial_transform=None,
                 temporal_transform=None):
        with open(annotation_path, 'rb') as f:
            self.data, self.idx_to_class = pickle.load(f)
        self.root_path = root_path
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform

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
        self.num_frames_in_clip = 91

        with open(annotation_path, "r") as f:
            if split == "train_1":
                fs = f.read()
                ''' The entrypoint of the bounding box data, bboxes are in the json file.
                    TODO: START HERE, look at json file, so how we can parse it differently to get boxes for each frame of video
                    debug annon, see what it consists of 
                    move on to line 159 of this file
                '''
                ann_dict = json.loads(fs) # this is where the annotations are <------------------------ maybe these have some more bboxes?
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
                            label = {'bounding_box': annon['box'], 'label': annon['action_ids']} # this needs to be a list of bboxes <--------- what is action_ids?
                            dp['labels'].append(label) # label is created here <-------------------------------------
                        self.data.append(dp)
            elif split == "val_1":
                fs = f.read()
                ann_dict = json.loads(fs)
                for video in ann_dict['db'].keys():
                    if split not in ann_dict['db'][video]['split_ids']:
                        continue

                    for frame in ann_dict['db'][video]['frames'].values():
                        if not frame['annotated'] or len(frame['annos']) == 0:
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
                        dp['labels'] = [] # <----------------------------- labels also created here (should be similar to above)
                        assert len(frame['annos']) > 0, frame['annotated']
                        for annon in frame['annos'].values():
                            label = {'bounding_box': annon['box'], 'label': annon['action_ids']}
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

    def __getitem__(self, index): # this might be more important actually 
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
                 split,
                 spatial_transform=None,
                 temporal_transform=None):
        self.data = [] # stores the other frames
        self.data_stride = [] # stores the stride'th frame
        self.fps = 12
        self.num_frames_in_clip = 91
        self.stride = 4 # distributed sampling, > stride means more unique data but less data points
        self.counter = 0

        with open(annotation_path, "r") as f:
            if split == "train_1":
                fs = f.read()
                ''' The entrypoint of the bounding box data, bboxes are in the json file.
                    TODO: START HERE, look at json file, so how we can parse it differently to get boxes for each frame of video
                    debug annon, see what it consists of 
                    move on to line 159 of this file
                '''
                ann_dict = json.loads(fs) 
                for video in ann_dict['db'].keys():
                    if split not in ann_dict['db'][video]['split_ids']:
                        continue

                    self.append_new_data(video, ann_dict)

            elif split == "val_1":
                fs = f.read()
                ann_dict = json.loads(fs)
                for video in ann_dict['db'].keys():
                    self.counter += 1
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
        stride_counter = 0
        for frame in ann_dict['db'][video]['frames'].values():
            dp = {}
            frame_id = int(frame['rgb_image_id'])
            dp['video'] = video
            dp['time'] = frame_id
            dp['midframe'] = frame_id
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
                    if frame_id + (self.num_frames_in_clip // 2) < ann_dict['db'][video]['numf'] \
                            and frame_id - (self.num_frames_in_clip // 2) >= 0:

                        dp['center_frame'] = len(self.data)
                        self.data_stride.append(dp)

            self.data.append(dp)

        return

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

    def __getitem__(self, index): # this might be more important actually 
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

    
class AVAmulticropDataLoader(AVADataLoader):
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
        
        output = {
            'clips': clips,
            'aug_info': aug_info,
            'filenames': filenames,
            'labels': labels,
            'mid_times': mid_times
        }
        return output


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
        batch_labels = [_['clip_labels'] for _ in batch]
        mid_times = [_['mid_time'] for _ in batch]
        
        output = {
            'clips': clips,
            'aug_info': aug_info,
            'filenames': filenames,
            'batch_labels': batch_labels,
            'mid_times': mid_times
        }
        return output
    
    
class AVAmulticrop(AVA):
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
