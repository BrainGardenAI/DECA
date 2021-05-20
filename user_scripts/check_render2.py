# -*- coding: utf-8 -*-
# conda env aphrodite: lg-avatar-pytorch3d (/home/developer/miniconda3/envs/lg-avatar-pytorch3d)

import os
import sys
import cv2
import json
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
from decalib.utils.rotation_converter import batch_euler2axis, deg2rad


on_server = True
path_prefix = "/" if on_server else "/home/s.orlova@dev.braingarden.ai/MOUNTED/aphrodite"
path_prefix = "/" if on_server else "/media/sveta/DATASTORE/WORK/aphrodite"


device = 'cuda'

# run DECA
scale = 2
target_size = 256
deca_cfg.model.uv_size = 256
deca_cfg.model.use_tex = False
deca_cfg.model.no_flametex_model = True
deca = DECA(config=deca_cfg, device=device)

frame_path = "disk/sdb1/avatars/dataset_TEST/LeaSeydoux/real/gP0TI4i84Hg/frames/00004.jpg"
param_path = "disk/sdb1/avatars/dataset_TEST/LeaSeydoux/real/gP0TI4i84Hg/params/flame_00004.npy"
obj_path = "disk/sdb1/avatars/dataset_TEST/LeaSeydoux/real/gP0TI4i84Hg/meshes_headrend/mesh_00004.obj"

in_array = np.load(os.path.join(path_prefix, param_path))
codedict = {}
for key in PARAM_DICT_ARR:
    start, stop, newshape, paramtype = PARAM_DICT_ARR[key]
    arr = np.reshape(in_array[start:stop], newshape=newshape)
    codedict[key] = torch.tensor(arr).type(paramtype).to(device) if key != "bbox" else arr
# read frame (image)
image = np.array(imread(os.path.join(path_prefix, frame_path)))
if len(image.shape) == 2:
    image = image[:, :, None].repeat(1, 1, 3)
if len(image.shape) == 3 and image.shape[2] > 3:
    image = image[:, :, :3]
h, w, _ = image.shape
# crop head with saved bbox coordinates
cx, cy, size, isFace = codedict["bbox"].tolist()
if not isFace:
    sys.exit()
src_pts = np.array([[cx - size / 2, cy - size / 2], [cx - size / 2, cy + size / 2], [cx + size / 2, cy - size / 2]])
dst_correction = np.array([[(scale-1)/2, (scale-1)/2]]) * target_size
DST_PTS = np.array([[0, 0], [0, target_size - 1], [target_size - 1, 0]]) + dst_correction
tform = estimate_transform('similarity', src_pts, DST_PTS)
dst_image = image / 255.
dst_image = warp(dst_image, tform.inverse, output_shape=(target_size*scale, target_size*scale))

# DEB
parpath = "/disk/sdb1/avatars/sveta/TEMP/DECA_cam/"
dst_image2  = np.clip(np.rint(dst_image*255), a_min=0, a_max=255).astype(np.uint8)
cv2.imwrite(os.path.join(parpath, "dst_img.jpg"), dst_image2)

dst_image = dst_image.transpose(2, 0, 1)
# now we have a dictionary for decoding
codedict["images"] = torch.tensor(dst_image).float().to(device)[None, ...]
codedict['cam'][:,0] /= 2

# build flame model with our image and parameters and TEXTURE
opdict, visdict = deca.decode(codedict)
opdict["light"] = None # !!!

# first save mesh in .obj file
temp_dir = os.path.join(os.path.abspath(os.getcwd()), "temp")
os.makedirs(temp_dir, exist_ok=False)
mesh_path = os.path.join(temp_dir, "temp_mesh.obj")
# save meshes (temporarily or permanently) and texture
deca.save_obj_my_format(filename=mesh_path, opdict=opdict, albedo_to_opdict=True)

# then get renders
results = deca.get_renderings(target_size=target_size*scale, uv_size=target_size*scale, mesh_file=mesh_path, opdict=opdict)
textured_image, normal_image, albedo_image = results
shutil.rmtree(temp_dir, ignore_errors=True)

if on_server:
    parpath = "/disk/sdb1/avatars/sveta/TEMP/DECA_cam/"
    cv2.imwrite(os.path.join(parpath, "frame.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(parpath, "textured.jpg"), textured_image)
    cv2.imwrite(os.path.join(parpath, "normal.jpg"), normal_image)
else:
    cv2.imshow('frame', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imshow('textured', textured_image)
    cv2.imshow('normal', normal_image)
    # cv2.imshow('albedo_image', albedo_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

