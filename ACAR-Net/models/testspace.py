import torch
import torch.nn as nn
from easydict import EasyDict
import yaml
import os

from necks import AVA_neck
from heads import AVA_head

dirname = os.path.dirname(os.path.abspath(__file__))

class TestModel(nn.Module):
    ''' Test Model, probes the head and neck of ACAR net for testing purposes
    '''
    def __init__(self, config):
        super(TestModel, self).__init__()
        self.config = config

        self.neck = AVA_neck(config.neck)
        self.head = AVA_head(config.head)

    def forward_head(self, o_b, o_n):
        i_h = {'features': o_b['features'], 'rois': o_n['rois'],
                   'num_rois': o_n['num_rois'], 'roi_ids': o_n['roi_ids'],
                   'sizes_before_padding': o_n['sizes_before_padding']}
        o_h = self.head(i_h)

        return o_h['outputs']

if __name__ == '__main__':
    # load config
    with open(os.path.join('/project/ACAR-Net/configs/ROAD/SLOWFAST_R50_ACAR_HR2O.yaml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    opt = EasyDict(config)

    # load model
    model = TestModel(opt.model)

    # load test data
    o_b = {
        'features': [torch.rand(8, 2048, 8, 16, 22), torch.rand(8, 256, 32, 16, 22)]
    }
    o_n = {
        'rois': torch.rand(42, 5),
        'num_rois': 42,
        'roi_ids': [0, 3, 9, 17, 25, 31, 33, 38, 42],
        'sizes_before_padding': [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
    }

    # eval
    print(model.forward_head(o_b, o_n).shape)
    