import math
import time

import torch
import torch.nn as nn
import torchvision

from .acar import HR2O_NL

__all__ = ['tube_acar']

class TubeACARHead(nn.Module):
    def __init__(self, width, roi_spatial=7, num_classes=60, dropout=0., bias=False,
                 reduce_dim=1024, hidden_dim=512, downsample='max2x2', depth=2, kernel_size=3):
        super(TubeACARHead, self).__init__()

        self.roi_spatial = roi_spatial
        self.roi_maxpool = nn.MaxPool2d(roi_spatial)
        
        # actor-context feature encoder
        self.conv_reduce = nn.Conv2d(width, reduce_dim, 1, bias=False)

        self.conv1 = nn.Conv2d(reduce_dim * 2, hidden_dim, 1, bias=False)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, bias=False)

        # down-sampling before HR2O
        assert downsample in ['none', 'max2x2']
        if downsample == 'none':
            self.downsample = nn.Identity()
        elif downsample == 'max2x2':
            self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # high-order relation reasoning operator (HR2O_NL)
        layers = []
        for _ in range(depth):
            layers.append(HR2O_NL(hidden_dim, kernel_size))
        self.hr2o = nn.Sequential(*layers)
        
        # classification
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(reduce_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim * 2, num_classes, bias=bias)

        if dropout > 0:
            self.dp = nn.Dropout(dropout)
        else:
            self.dp = None

    def forward(self, data):
        ''' ACAR HEAD: takes in data from the backbone and neck, produces action predictions through HR2O
            Coming in rois must be of shape [num_rois, n_fast_frames, 5].
            When no roi exists, this code expects an roi of [batch_num, 1, 1, 1, 1] or
            in other words, an invalid roi. This creates a tensor of zeros.
            
            data:
                features: slow and fast features, [0] is slow, [1] is fast
                rois: all the rois we are concerned with 
                    (each label has 32 bboxes corrosponding to each fast frame)
                num_rois: total number of rois
                roi_ids: where each batch of rois end in the list of rois
                sizes_before_padding: image size before padding
            returns:
                outputs: class predictions
        '''
        if not isinstance(data['features'], list):
            feats = [data['features']]
        else:
            feats = data['features']
        
        # batch size, channels, # of frames, height, width
        # slow feature shape
        B_s, C_s, N_s, H_s, W_s = feats[0].shape #(8, 2048, 8, 16, 22)
        # fast feature shape
        B_f, C_f, N_f, H_f, W_f = feats[1].shape #(8, 256, 32, 16, 22)
        roi_slow_feats = []
        roi_fast_feats = []
        roi_slow_feats_nonzero = []
        roi_fast_feats_nonzero = [] 

        # for each temporal fast encoding, roi align
        alpha = int(N_f / N_s)
        for idx in range(N_f):
            if (idx + 1) % alpha == 0: # for each temporal slow encoding
                f_s = feats[0][:, :, int(idx / alpha)] # (8, 2048, 16, 22)
                rois = data['rois'][:, idx] # roi for every alpha frame, (num_rois, 5)
                
                roi_slow_feats_nonzero.append(~(rois[:, 1:]==1).all(1)) # boolean mask for filtering invalid rois, (num_rois)
                roi_slow_feats.append(self.head_roi_align(rois.clone(), f_s, H_s, W_s)) # (2048, 7, 7) each

            f_f = feats[1][:, :, idx] # (8, 256, 16, 22)
            rois = data['rois'][:, idx] # roi for every frame, (num_rois, 5)

            roi_fast_feats_nonzero.append(~(rois[:, 1:]==1).all(1)) # boolean mask for filtering invalid rois ()
            roi_fast_feats.append(self.head_roi_align(rois.clone(), f_f, H_s, W_s)) # (256, 7, 7) each
            
        # stack pooled fast and slow roi alignments
        # these are of shape [num_rois, depth, n_slow_or_fast_frames, roi_spatial, roi_spatial]
        roi_slow_feats = torch.stack(roi_slow_feats, dim=2) # (num_rois, 2048, 8, 7, 7)
        roi_fast_feats = torch.stack(roi_fast_feats, dim=2) # (num_rois, 256, 32, 7, 7)
        # these are of shape [num_rois, n_slow_or_fast_frames]
        roi_slow_feats_nonzero = torch.stack(roi_slow_feats_nonzero, dim=1) #(num_rois, 8)
        roi_fast_feats_nonzero = torch.stack(roi_fast_feats_nonzero, dim=1) #(num_rois, 32)

        # filter out invalid rois and avg pool
        # avg pooling is done over the frames in the clip, which means it is done temporally
        # it seems that this throws away anything that may be learned from the sequence of activation maps
        # corresponding to each frame in the clips.
        roi_slow_feats = [nn.AdaptiveAvgPool3d((1, self.roi_spatial, self.roi_spatial))(roi_slow_feats[idx, :, s_mask]) 
            for idx, s_mask in enumerate(roi_slow_feats_nonzero)]
        roi_fast_feats = [nn.AdaptiveAvgPool3d((1, self.roi_spatial, self.roi_spatial))(roi_fast_feats[idx, :, f_mask]) 
            for idx, f_mask in enumerate(roi_fast_feats_nonzero)]

        # stack pooled fast and slow roi alignments, squeeze frame dim
        roi_slow_feats = torch.stack(roi_slow_feats, dim=0).squeeze(dim=2) # (num_rois, 2048, 7, 7)
        roi_fast_feats = torch.stack(roi_fast_feats, dim=0).squeeze(dim=2) # (num_rois, 256, 7, 7)

        # concatenate and reduce the slow and fast roi_feats
        roi_feats = torch.cat([roi_slow_feats, roi_fast_feats], dim=1) # (num_rois, 2304, 7, 7)
        # lots of learnable parameters here
        # conv reduce has a (1,1) kernel. The number of output channels is
        # set to be 1024, so there are 1024 filters, which are applied to each
        # input plane (channel). As each filter corresponds to one output plane, 
        # each input plane must have different weights associated with each filter.
        # since there are 2304 input planes (256 + 2048), then there are 
        # 2048 * 1024 = 2359296 learnable parameters in this line.
        roi_feats = self.conv_reduce(roi_feats)
        # now the shape is (num_rois, 1024, 7, 7)

        # roi maxpool
        roi_feats = self.roi_maxpool(roi_feats).view(data['num_rois'], -1)
        # now the shape is (num_rois, 1024)

        # temporal average pooling for later tiling
        # requires all features have the same temporal dimensions
        # Average pooling is done over the clips, like what is done with the rois.
        feats = [nn.AdaptiveAvgPool3d((1, H_s, W_s))(f).view(-1, f.shape[1], H_s, W_s) for f in feats] # (8, 2048, 16, 22)
        feats = torch.cat(feats, dim=1) # (2304, 16, 22)
        # a lot of params here as well.
        feats = self.conv_reduce(feats) # (1024, 16, 22)
        
        # downstream tiling
        roi_ids = data['roi_ids']
        sizes_before_padding = data['sizes_before_padding']
        high_order_feats = []
        for idx in range(feats.shape[0]):  # iterate over mini-batch
            # n_rois is the number labels/rois in this key frame 
            n_rois = roi_ids[idx+1] - roi_ids[idx]
            if n_rois == 0:
                continue
            
            eff_h, eff_w = math.ceil(H_s * sizes_before_padding[idx][1]), math.ceil(W_s * sizes_before_padding[idx][0])
            bg_feats = feats[idx][:, :eff_h, :eff_w] # (1024, 16, 22)
            bg_feats = bg_feats.unsqueeze(0).repeat((n_rois, 1, 1, 1)) # (n_rois, 1024, 16, 22)
            actor_feats = roi_feats[roi_ids[idx]:roi_ids[idx+1]] # (n_rois, 1024)
            tiled_actor_feats = actor_feats.unsqueeze(2).unsqueeze(2).expand_as(bg_feats) # (n_rois, 1024, 16, 22)
            interact_feats = torch.cat([bg_feats, tiled_actor_feats], dim=1) # (2, 2048, 16, 22)

            interact_feats = self.conv1(interact_feats) # (2, 512, 16, 22)
            interact_feats = nn.functional.relu(interact_feats) 
            interact_feats = self.conv2(interact_feats) # (2, 512, 14, 20)
            interact_feats = nn.functional.relu(interact_feats)

            interact_feats = self.downsample(interact_feats) # (2, 512, 7, 10)
            # hr2o input and output dimensions are equal
            interact_feats = self.hr2o(interact_feats) # (2, 512, 7, 10)
            interact_feats = self.gap(interact_feats) # (2, 512, 1, 1)
            high_order_feats.append(interact_feats)

        high_order_feats = torch.cat(high_order_feats, dim=0).view(data['num_rois'], -1) # (66, 512)
        
        outputs = self.fc1(roi_feats) # (num_rois, 512)
        outputs = nn.functional.relu(outputs)
        outputs = torch.cat([outputs, high_order_feats], dim=1) # (num_rois, 1024)

        if self.dp is not None:
            outputs = self.dp(outputs)
        outputs = self.fc2(outputs) # (num_rois, 22)

        return {'outputs': outputs}

    def head_roi_align(self, rois, frame, h, w):
        rois[:, 1] = rois[:, 1] * w
        rois[:, 2] = rois[:, 2] * h
        rois[:, 3] = rois[:, 3] * w
        rois[:, 4] = rois[:, 4] * h

        rois = rois.detach()

        return torchvision.ops.roi_align(frame, rois, (self.roi_spatial, self.roi_spatial))

def tube_acar(**kwargs):
    model = TubeACARHead(**kwargs)
    return model
