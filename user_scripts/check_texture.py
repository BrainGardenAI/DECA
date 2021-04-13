"""
That's just a script for checking some ideas about texture format
"""

import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F

from decalib.datasets.constants import *

mean_tex_path = "/home/s.orlova@dev.braingarden.ai/Projects/DECA/data/mean_texture_ORIG.jpg"
tex_path = "/home/s.orlova@dev.braingarden.ai/Projects/DECA/data/FLAME_albedo_from_BFM.npz"
texture_default = cv2.imread(mean_tex_path)
texture_default = cv2.resize(texture_default, (256, 256), cv2.INTER_CUBIC)

bfm_file = np.load(tex_path)
texture_mean = bfm_file['MU']
texture_basis = bfm_file['PC']

texture_mean = texture_mean.reshape(1, -1)
texture_basis = texture_basis.reshape(-1, 199)

n_tex = 50
num_components = texture_basis.shape[1]
texture_mean = torch.from_numpy(texture_mean).float()[None,...]
texture_basis = torch.from_numpy(texture_basis[:,:n_tex]).float()[None,...]


param_path = "/home/s.orlova@dev.braingarden.ai/Data/From_YT2_processed/NormanReedus/virtual/0-NormanReedus/params/flame_00000.npy"
in_array = np.load(param_path)
codedict = {}
for key in PARAM_DICT_ARR:
    start, stop, newshape, paramtype = PARAM_DICT_ARR[key]
    arr = np.reshape(in_array[start:stop], newshape=newshape)
    codedict[key] = torch.tensor(arr).type(paramtype) if key != "bbox" else arr
texcode = codedict['tex']

"""
texcode: [batchsize, n_tex]
texture: [bz, 3, 256, 256], range: 0-1
"""
texture = texture_mean + (texture_basis*texcode[:,None,:]).sum(-1)
texture = texture.reshape(texcode.shape[0], 512, 512, 3).permute(0,3,1,2)
texture = F.interpolate(texture, [256, 256])
#texture = texture[:,[2,1,0], :,:] # bgr -> rgb

texture = texture[0].permute(1,2,0)
texture = np.array(texture)
texture = np.rint(texture*255).astype(np.uint8)

cv2.imshow('tex', texture)
cv2.imshow('tex_def', texture_default)
cv2.waitKey()
cv2.destroyAllWindows()

print("")
