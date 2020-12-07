import os
import cv2

from data import common

import numpy as np
import imageio

import torch
import torch.utils.data as data

from IPython.core import debugger 
breakpoint = debugger.set_trace


class Demo(data.Dataset):
    def __init__(self, args, name='Demo', train=False, benchmark=False):
        
        self.args = args
        self.name = name
        self.scale = args.scale
        self.idx_scale = 0
        self.train = False
        self.benchmark = benchmark

        self.filelist = []
        for f in os.listdir(args.dir_demo):
            if f.find('.png') >= 0 or f.find('.jpg') >= 0 or f.find('.hdr'):
                self.filelist.append(os.path.join(args.dir_demo, f))
        self.filelist.sort()

    def __getitem__(self, idx):
        
        filename = os.path.splitext(os.path.basename(self.filelist[idx]))[0]
        

        if '.hdr' in self.filelist[idx]:
            lr = cv2.imread(self.filelist[idx],  cv2.IMREAD_ANYDEPTH)
            lr = cv2.cvtColor(lr,cv2.COLOR_BGR2RGB)
            self.args.max_val = np.max(lr)
            #self.args.rgb_range = np.max(lr)
            lr = lr*(255/self.args.max_val)
        else:
            lr = imageio.imread(self.filelist[idx])
            #self.args.rgb_range = 255
            #breakpoint()
        #print(lr, "is of type", type(lr))
        
        breakpoint()
        lr, = common.set_channel(lr, n_channels=self.args.n_colors)

        
        lr_t, = common.np2Tensor(lr, rgb_range=self.args.rgb_range)

        #print(lr_t, "is of type", type(lr))
        #breakpoint()

        return lr_t, -1, filename

    def __len__(self):
        return len(self.filelist)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

