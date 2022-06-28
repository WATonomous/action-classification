import math
import time

import torch
import torch.nn as nn
import torchvision

__all__ = ['acar']


class HR2O_NL(nn.Module):
    def __init__(self, hidden_dim=512, kernel_size=3):
        super(HR2O_NL, self).__init__()

        self.hidden_dim = hidden_dim

        padding = kernel_size // 2
        self.conv_q = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_k = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_v = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.norm = nn.GroupNorm(1, hidden_dim, affine=True)
        self.dp = nn.Dropout(0.2)

    def forward(self, x):
        query = self.conv_q(x).unsqueeze(1)
        key = self.conv_k(x).unsqueeze(0)
        att = (query * key).sum(2) / (self.hidden_dim ** 0.5)
        att = nn.Softmax(dim=1)(att)
        value = self.conv_v(x)
        virt_feats = (att.unsqueeze(2) * value).sum(1)

        virt_feats = self.norm(virt_feats)
        virt_feats = nn.functional.relu(virt_feats)
        virt_feats = self.conv(virt_feats)
        virt_feats = self.dp(virt_feats)
        
        x = x + virt_feats
        return x


class ACARHead(nn.Module):
    def __init__(self, width, roi_spatial=7, num_classes=60, dropout=0., bias=False,
                 reduce_dim=1024, hidden_dim=512, downsample='max2x2', depth=2, kernel_size=3):
        super(ACARHead, self).__init__()

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
        
        B_s, C_s, N_s, H_s, W_s = feats[0].shape 
        B_f, C_f, N_f, H_f, W_f = feats[1].shape
        roi_slow_feats = []
        roi_fast_feats = []
        roi_slow_feats_nonzero = []
        roi_fast_feats_nonzero = [] 

        # for each temporal fast encoding, roi align
        alpha = int(N_f / N_s)
        for idx in range(N_f):

            if (idx + 1) % alpha == 0: # for each temporal slow encoding
                f_s = feats[0][:, :, int(idx / alpha)]
                rois = data['rois'][:, idx].detach() # roi for every alpha frame
                
                roi_slow_feats_nonzero.append(~(rois[:, 1:]==1).all(1)) # mask for filtering invalid rois
                roi_slow_feats.append(self.head_roi_align(rois.clone(), f_s, H_s, W_s))

            f_f = feats[1][:, :, idx]
            rois = data['rois'][:, idx].detach() # roi for every frame

            roi_fast_feats_nonzero.append(~(rois[:, 1:]==1).all(1)) # mask for filtering invalid rois
            roi_fast_feats.append(self.head_roi_align(rois.clone(), f_f, H_s, W_s))
            
        # stack pooled fast and slow roi alignments
        # these are of shape [num_rois, depth, n_slow_or_fast_frames, roi_spatial, roi_spatial]
        roi_slow_feats = torch.stack(roi_slow_feats, dim=2) 
        roi_fast_feats = torch.stack(roi_fast_feats, dim=2) 
        # these are of shape [num_rois, n_slow_or_fast_frames]
        roi_slow_feats_nonzero = torch.stack(roi_slow_feats_nonzero, dim=1)
        roi_fast_feats_nonzero = torch.stack(roi_fast_feats_nonzero, dim=1)

        # filter out invalid rois and avg pool
        roi_slow_feats = [nn.AdaptiveAvgPool3d((1, self.roi_spatial, self.roi_spatial))(roi_slow_feats[idx, :, s_mask]) 
            for idx, s_mask in enumerate(roi_slow_feats_nonzero)]
        roi_fast_feats = [nn.AdaptiveAvgPool3d((1, self.roi_spatial, self.roi_spatial))(roi_fast_feats[idx, :, f_mask]) 
            for idx, f_mask in enumerate(roi_fast_feats_nonzero)]

        # stack pooled fast and slow roi alignments, squeeze frame dim
        roi_slow_feats = torch.stack(roi_slow_feats, dim=0).squeeze(dim=2)
        roi_fast_feats = torch.stack(roi_fast_feats, dim=0).squeeze(dim=2)

        # concatenate and reduce the slow and fast roi_feats
        roi_feats = torch.cat([roi_slow_feats, roi_fast_feats], dim=1)
        roi_feats = self.conv_reduce(roi_feats)

        # roi maxpool
        roi_feats = self.roi_maxpool(roi_feats).view(data['num_rois'], -1)

        # temporal average pooling for later tiling
        # requires all features have the same spatial dimensions
        feats = [nn.AdaptiveAvgPool3d((1, H_s, W_s))(f).view(-1, f.shape[1], H_s, W_s) for f in feats]
        feats = torch.cat(feats, dim=1)
        feats = self.conv_reduce(feats)
        
        # downstream tiling
        roi_ids = data['roi_ids']
        sizes_before_padding = data['sizes_before_padding']
        high_order_feats = []
        for idx in range(feats.shape[0]):  # iterate over mini-batch
            n_rois = roi_ids[idx+1] - roi_ids[idx]
            if n_rois == 0:
                continue
            
            eff_h, eff_w = math.ceil(H_s * sizes_before_padding[idx][1]), math.ceil(W_s * sizes_before_padding[idx][0])
            bg_feats = feats[idx][:, :eff_h, :eff_w]
            bg_feats = bg_feats.unsqueeze(0).repeat((n_rois, 1, 1, 1))
            actor_feats = roi_feats[roi_ids[idx]:roi_ids[idx+1]]
            tiled_actor_feats = actor_feats.unsqueeze(2).unsqueeze(2).expand_as(bg_feats)
            interact_feats = torch.cat([bg_feats, tiled_actor_feats], dim=1)

            interact_feats = self.conv1(interact_feats)
            interact_feats = nn.functional.relu(interact_feats)
            interact_feats = self.conv2(interact_feats)
            interact_feats = nn.functional.relu(interact_feats)

            interact_feats = self.downsample(interact_feats)

            interact_feats = self.hr2o(interact_feats)
            interact_feats = self.gap(interact_feats)
            high_order_feats.append(interact_feats)

        high_order_feats = torch.cat(high_order_feats, dim=0).view(data['num_rois'], -1)
        
        outputs = self.fc1(roi_feats)
        outputs = nn.functional.relu(outputs)
        outputs = torch.cat([outputs, high_order_feats], dim=1)

        if self.dp is not None:
            outputs = self.dp(outputs)
        outputs = self.fc2(outputs)

        return {'outputs': outputs}

    def head_roi_align(self, rois, frame, h, w):
        rois[:, 1] = rois[:, 1] * w
        rois[:, 2] = rois[:, 2] * h
        rois[:, 3] = rois[:, 3] * w
        rois[:, 4] = rois[:, 4] * h

        rois = rois.detach()

        return torchvision.ops.roi_align(frame, rois, (self.roi_spatial, self.roi_spatial))

def acar(**kwargs):
    model = ACARHead(**kwargs)
    return model
