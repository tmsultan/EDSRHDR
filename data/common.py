import random

import numpy as np
import skimage.color as sc

from IPython.core import debugger 
breakpoint = debugger.set_trace


import torch

def get_patch(*args, patch_size=96, scale=2, multi=False, input_large=False):
    ih, iw = args[0].shape[:2]

    if not input_large:
        p = scale if multi else 1
        tp = p * patch_size
        ip = tp // scale
    else:
        tp = patch_size
        ip = patch_size

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    if not input_large:
        tx, ty = scale * ix, scale * iy
    else:
        tx, ty = ix, iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    return ret

def set_channel(*args, n_channels=3):
    def _set_channel(img):

        
        # If image is 2D, add a third dimension for color
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
       
        
        # Shape gives number of values in each dimension,
        # c therefore is the 3rd element (indexed from 0), so refers to the number of channels
        c = img.shape[2]

        # If we want number of channels to be 1, but have 3 channels (i.e. rgb image)
            # Expand to a 4th dimension
            # Convert rgb to rgb2ycbcr
            # Take the 0th channel and expand it??? What is 2 about
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)
        
        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=255):
#def np2Tensor(*args, rgb_range=args.rgb_range):
    def _np2Tensor(img):

        #print(type(img))
        
        # Tranpose color channels - cast as float 32
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)), dtype=np.float32)

        
        # Create torch tensor
        tensor = torch.from_numpy(np_transpose).float()

        
        # Should update rgb_range - takes and image and scales by rgb_range
        # Currently not using rgb_range
        tensor.mul_(rgb_range / 255)

        #breakpoint()

        return tensor

    return [_np2Tensor(a) for a in args]

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(a) for a in args]

