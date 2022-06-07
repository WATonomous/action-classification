from PIL import Image
import os
import pickle
import json
import numpy as np
import io
from iopath.common.file_io import g_pathmgr

import torch
import torch.nn.functional as F

class Test():
    def __init__(self):
        self.list = ['a', 'b', 'c', 'd', 'e']

    def __getitem__(self, index):
        if index == 2:
            return None 
        else:
            return self.list[index]


if __name__ == '__main__':
    print(list(range(3, 5)))
    
    
