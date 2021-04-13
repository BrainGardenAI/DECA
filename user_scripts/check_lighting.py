# -*- coding: utf-8 -*-
# conda env aphrodite: lg-avatar-pytorch3d (/home/developer/miniconda3/envs/lg-avatar-pytorch3d)

import os
import sys
import cv2
import shutil
import argparse
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import estimate_transform, warp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.utils import util
from decalib.datasets import datasets
from decalib.datasets.constants import *
from decalib.utils.config import cfg as deca_cfg


device = 'cuda'

# run DECA
deca_cfg.model.use_tex = False
deca_cfg.model.no_flametex_model = True
deca = DECA(config=deca_cfg, device=device)

frame_path = "/home/s.orlova@dev.braingarden.ai/Data/PAIRS/sets/2/frames/00066.jpg"
param_path = "/home/s.orlova@dev.braingarden.ai/Data/PAIRS/sets/2/params/flame_00066.npy"
in_array = np.load(param_path)
codedict = {}
for key in PARAM_DICT_ARR:
    start, stop, newshape, paramtype = PARAM_DICT_ARR[key]
    arr = np.reshape(in_array[start:stop], newshape=newshape)
    codedict[key] = torch.tensor(arr).type(paramtype).to(device) if key != "bbox" else arr
# read frame (image)
image = np.array(imread(frame_path))
if len(image.shape) == 2:
    image = image[:, :, None].repeat(1, 1, 3)
if len(image.shape) == 3 and image.shape[2] > 3:
    image = image[:, :, :3]
h, w, _ = image.shape
# crop head with saved bbox coordinates
cx, cy, size, isFace = codedict["bbox"].tolist()
if not isFace:
    sys.exit()
target_size = 224
src_pts = np.array([[cx - size / 2, cy - size / 2], [cx - size / 2, cy + size / 2], [cx + size / 2, cy - size / 2]])
DST_PTS = np.array([[0, 0], [0, target_size - 1], [target_size - 1, 0]])
tform = estimate_transform('similarity', src_pts, DST_PTS)
dst_image = image / 255.
dst_image = warp(dst_image, tform.inverse, output_shape=(target_size, target_size))
dst_image = dst_image.transpose(2, 0, 1)
# now we have a dictionary for decoding
codedict["images"] = torch.tensor(dst_image).float().to(device)[None, ...]

# build flame model with our image and parameters and TEXTURE
opdict, visdict = deca.decode(codedict)
opdict["light"] = codedict["light"] # !!!
# first save mesh in .obj file
temp_dir = os.path.join(os.path.abspath(os.getcwd()), "temp")
os.makedirs(temp_dir, exist_ok=False)
mesh_path = os.path.join(temp_dir, "temp_mesh.obj")
# save meshes (temporarily or permanently) and texture
deca.save_obj_my_format(filename=mesh_path, opdict=opdict, albedo_to_opdict=True)

# then get renders
results = deca.get_renderings(target_size=target_size, mesh_file=mesh_path, opdict=opdict)
textured_image, normal_image, albedo_image = results
shutil.rmtree(temp_dir, ignore_errors=True)

cv2.imshow('frame', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.imshow('textured', textured_image)
cv2.imshow('normal', normal_image)
# cv2.imshow('albedo_image', albedo_image)
cv2.waitKey()
cv2.destroyAllWindows()
