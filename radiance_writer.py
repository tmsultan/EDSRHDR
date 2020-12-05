# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 11:17:21 2018

@author: ingle, yuhao
"""
import numpy as np
import cv2

from IPython.core import debugger 
breakpoint = debugger.set_trace


def radiance_writer(image, fname):

    print(image.shape[0])

    image = image.numpy()
    

    """image should be a 3D matrix of RGB values in floating point numbers for each pixel"""
    brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])

    # Return an array of zeros with the same shape and type as a given array.
    mantissa = np.zeros_like(brightest)
    exponent = np.zeros_like(brightest)

    # Store to higher precision
    mantissa, exponent = np.frexp(brightest, mantissa, exponent)

    # Mantissa: Exponent Product - scale to 256 
    # Divide by zero --> can create issues
    scaled_mantissa = mantissa * 256.0 / brightest

    # Create zeros size - Image H * W * 4 Channels
    rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)

    # around: Evenly round to the given number of decimals.
    rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
    rgbe[...,3] = np.around(exponent + 128)
    
    
    rgbe[rgbe>255] = 255
    rgbe[rgbe<0] = 0
    rgbe = np.array(rgbe, dtype=np.uint8)

    f = open(fname, "wb")
    f.write("#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n".encode())
    f.write("-Y {0} +X {1}\n".format(image.shape[0], image.shape[1]).encode())
    
    

    rgbe.flatten().tofile(f)
    f.close()

    #breakpoint()
    # print(fname, 'written.')




