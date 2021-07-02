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
import decalib.datasets.detectors as detectors
from decalib.utils.config import cfg as deca_cfg


def image_preprocessing(imagepath, crop_size, face_too_close=False):
    image = np.array(imread(imagepath))
    if len(image.shape) == 2:
        image = image[:, :, None].repeat(1, 1, 3)
    if len(image.shape) == 3 and image.shape[2] > 3:
        image = image[:, :, :3]

    isFace = False
    h, w, _ = image.shape

    if face_too_close:
        new_h = h*2.5
        pad_t = (new_h - h) // 2
        image = cv2.copyMakeBorder(image, top=pad_t, bottom=new_h-h-pad_t, left=0, right=0, borderType=cv2.BORDER_CONSTANT)

    bbox, bbox_type, kpt = detectors.FAN().run(image, return_kpt=True)
    # if detector doesn't find head bbox, lie that the whole image is bbox (>_>)
    if len(bbox) < 4:
        # print('no face detected! run original image')
        left = 0
        top = 0
        right = h - 1
        bottom = w - 1
    # if detector found the face we wanted
    else:
        left = bbox[0]
        top = bbox[1]
        right = bbox[2]
        bottom = bbox[3]
        isFace = True
    # from bbox2point for type kpt68
    if bbox_type == "kpt68":
        old_size = (right - left + bottom - top) / 2 * 1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    elif bbox_type == "no_detection":
        old_size = 0
        center = np.array([0, 0])
    else:
        print("bbox_type is wrong - {}".format(bbox_type))
        exit()

    size = int(old_size * 1.25)
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])

    DST_PTS = np.array([[0, 0], [0, crop_size - 1], [crop_size - 1, 0]])

    image = image / 255.
    tform = estimate_transform('similarity', src_pts, DST_PTS)
    dst_image = warp(image, tform.inverse, output_shape=(crop_size, crop_size))

    bbox = (center[0], center[1], size, True) if isFace else (-1, -1, -1, False)
    if face_too_close and isFace:
        bbox = (center[0], center[1]-pad_t, size, True)
        kpt += np.array([[0, -pad_t]])
    dst_image = dst_image.transpose(2, 0, 1)
    return {'image': torch.tensor(dst_image).float(),
            'bbox': bbox,
            'isFace': isFace,
            'kpt': kpt}

def generate_data(subset, video):

    # make a list of all frame directories that we'll process
    inp_framedir = os.path.join(frames_root, subset, domain, video, "frames")
    frames = sorted([f for f in os.listdir(inp_framedir) if os.path.splitext(f)[1] in FRAME_SUFFIX])
    L = len(frames)

    # decide what texture renders and meshes we are saving now, and set corresponding DECA config parameters
    if texture_type == "head_mask":
        DIR_meshes = DIR_MESHES_HEADMASK
        DIR_renders_tex = DIR_RENDERS_TEX_HEADMASK
        DIR_albedos = DIR_RENDERS_ALB_HEADMASK
        DIR_shadows = DIR_RENDERS_SHA_HEADMASK
        FILE_renders_tex = FILE_RENDERS_TEX_HEADMASK
        FILE_meshes_temp = FILE_MESHES_COARSE_TEMP_HEADMASK
        FILE_albedos = FILE_RENDERS_ALB_HEADMASK
        FILE_shadows = FILE_RENDERS_SHA_HEADMASK
        deca_cfg.model.use_tex = False
        deca_cfg.model.no_flametex_model = True # False is ok too, it doesn't matter here
    elif texture_type == "head_render":
        DIR_meshes = DIR_MESHES_HEADREND
        DIR_renders_tex = DIR_RENDERS_TEX_HEADREND
        DIR_albedos = DIR_RENDERS_ALB_HEADREND
        DIR_shadows = DIR_RENDERS_SHA_HEADREND
        FILE_renders_tex = FILE_RENDERS_TEX_HEADREND
        FILE_meshes_temp = FILE_MESHES_COARSE_TEMP_HEADREND
        FILE_albedos = FILE_RENDERS_ALB_HEADREND
        FILE_shadows = FILE_RENDERS_SHA_HEADREND
        deca_cfg.model.use_tex = True
        deca_cfg.model.no_flametex_model = False
    else:
        DIR_meshes = DIR_MESHES
        DIR_renders_tex = DIR_RENDERS_TEX
        DIR_albedos = DIR_RENDERS_ALB
        DIR_shadows = DIR_RENDERS_SHA
        FILE_renders_tex = FILE_RENDERS_TEX
        FILE_meshes_temp = FILE_MESHES_COARSE_TEMP
        FILE_albedos = FILE_RENDERS_ALB
        FILE_shadows = FILE_RENDERS_SHA
        deca_cfg.model.use_tex = True
        deca_cfg.model.no_flametex_model = True

    # run DECA
    deca = DECA(config=deca_cfg, device=device)

    # loop over directories with frames extracted from video
    print("start iterating..")
    for frame in tqdm(frames):
        frame_path = os.path.join(inp_framedir, frame)
        frame_id = os.path.splitext(frame)[0]

        renddir = os.path.join(renders_out_root, subset, domain, video)
        normal_path = os.path.join(renddir, DIR_RENDERS_NORM, FILE_FRAME_TEMP.format(frame_id))
        pos_mask_path = os.path.join(renddir, DIR_RENDERS_POS, FILE_FRAME_TEMP.format(frame_id))
        texture_path = os.path.join(renddir, DIR_renders_tex, FILE_FRAME_TEMP.format(frame_id))
        albedo_path = os.path.join(renddir, DIR_albedos, FILE_FRAME_TEMP.format(frame_id))
        shading_path = os.path.join(renddir, DIR_shadows, FILE_FRAME_TEMP.format(frame_id))

        rendtypes = ("normals", "pos_mask", "texture", "shading", "albedo")
        paths = (normal_path, pos_mask_path, texture_path, shading_path, albedo_path)

        if not any([(rt in what_to_save) and not os.path.exists(p) for rt, p in zip(rendtypes, paths)]):
            continue

        # DECA ENCODE
        # get param dict
        preprocessed_data = image_preprocessing(frame_path,
                                                crop_size=image_crop_size,
                                                face_too_close=True if video == "6-LeaSeydoux" else False)
        if not preprocessed_data['isFace']:
            continue
        images = preprocessed_data['image'].to(device)[None, ...]
        codedict = deca.encode(images)

        # prepare original frame to be saved in codedict
        # crop head with saved bbox coordinates and given scale
        image = np.array(imread(frame_path))
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(1, 1, 3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]
        H, W, _ = image.shape
        cx, cy, size, isFace = preprocessed_data['bbox']
        if not isFace:
            continue

        # DEB
        """
        outdir = "/disk/sdb1/avatars/extra_repo/DECA/temp"
        cv2.rectangle(image, pt1=(int(cx - size / 2), int(cy - size / 2)), pt2=(int(cx + size / 2), int(cy + size / 2)), color=(0,255,0), thickness=4)
        #print("kpt:", preprocessed_data["kpt"][:5])
        #exit()
        for pt in preprocessed_data["kpt"]:
            cv2.circle(image, center=(int(pt[0]), int(pt[1])), radius = 4, color = (128, 0, 128), thickness=-1)
        cv2.imwrite(os.path.join(outdir, "Lea6_{}.jpg".format(frame_id)), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        exit()
        """

        src_pts = np.array([[cx - size / 2, cy - size / 2], [cx - size / 2, cy + size / 2], [cx + size / 2, cy - size / 2]])
        dst_correction = np.array([[(scale - 1) / 2, (scale - 1) / 2]]) * target_size
        DST_PTS = np.array([[0, 0], [0, target_size - 1], [target_size - 1, 0]]) + dst_correction
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        dst_image = image / 255.
        dst_image = warp(dst_image, tform.inverse, output_shape=(target_size*scale, target_size*scale))
        dst_image = dst_image.transpose(2, 0, 1)
        # now we have a dictionary for decoding
        codedict["images"] = torch.tensor(dst_image).float().to(device)[None, ...]
        codedict['cam'][:, 0] /= scale

        # DECODING
        # build flame model with our image and parameters and TEXTURE
        print("decoding..")
        opdict, visdict = deca.decode(codedict)
        opdict["light"] = codedict["light"]

        # save meshes (temporarily or permanently) and texture
        temp_dir = os.path.join(os.path.abspath(os.getcwd()), "temp")
        os.makedirs(temp_dir, exist_ok=False)
        mesh_path = os.path.join(temp_dir, "temp_mesh.obj")
        deca.save_obj_my_format(filename=mesh_path, opdict=opdict, albedo_to_opdict=True)

        # get renders
        results = deca.get_renderings(target_size=target_size*scale, uv_size=target_size*scale, mesh_file=mesh_path,
                                      opdict=opdict, more_data=True)
        textured_image, normal_image, albedo_image, shading_image, pos_mask = results

        shutil.rmtree(temp_dir, ignore_errors=True)

        # align and resize renders
        hsize = size*scale/2
        # three points of the square face bbox on the frame
        dst_pts = np.array([[cx - hsize, cy - hsize], [cx - hsize, cy + hsize], [cx + hsize, cy - hsize]])
        # the same points on renders
        src_pts = np.array([[0, 0], [0, target_size*scale - 1], [target_size*scale - 1, 0]])

        # resize renders
        tform = estimate_transform('similarity', src_pts, dst_pts)
        textured_image = warp(textured_image / 255., tform.inverse, output_shape=(H, W))
        normal_image = warp(normal_image / 255., tform.inverse, output_shape=(H, W), cval=0.502)  # we need 128, not 127
        albedo_image = warp(albedo_image / 255., tform.inverse, output_shape=(H, W))
        shading_image = warp(shading_image / 255., tform.inverse, output_shape=(H, W), cval=0.797)  # we need 203
        pos_mask = warp(pos_mask / 255., tform.inverse, output_shape=(H, W))
        textured_image = np.clip(textured_image * 255, a_min=0, a_max=255).astype(np.uint8)
        normal_image = np.clip(normal_image * 255, a_min=0, a_max=255).astype(np.uint8)
        albedo_image = np.clip(albedo_image * 255, a_min=0, a_max=255).astype(np.uint8)
        shading_image = np.clip(shading_image * 255, a_min=0, a_max=255).astype(np.uint8)
        pos_mask = np.clip(pos_mask * 255, a_min=0, a_max=255).astype(np.uint8)

        # remove all background of a shading image using pos_mask
        kernel = np.ones((5, 5), np.uint8)
        shading_image = np.rint(shading_image * (cv2.dilate(pos_mask, kernel, iterations=1)/255.)).astype(np.uint8)

        #print("\n\nshading background value: {}".format(shading_image[3,3]))
        #exit()

        # save renders
        if "normals" in what_to_save:
            os.makedirs(os.path.split(normal_path)[0], exist_ok=True)
            cv2.imwrite(normal_path, normal_image)
        if "pos_mask" in what_to_save:
            os.makedirs(os.path.split(pos_mask_path)[0], exist_ok=True)
            cv2.imwrite(pos_mask_path, pos_mask)
        if "texture" in what_to_save:
            os.makedirs(os.path.split(texture_path)[0], exist_ok=True)
            cv2.imwrite(texture_path, textured_image)
        if "shading" in what_to_save:
            os.makedirs(os.path.split(shading_path)[0], exist_ok=True)
            cv2.imwrite(shading_path, shading_image)
        if "albedo" in what_to_save:
            os.makedirs(os.path.split(albedo_path)[0], exist_ok=True)
            cv2.imwrite(albedo_path, albedo_image)

        if do_show:
            cv2.imshow('frame', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imshow('textured', textured_image)
            cv2.imshow('normal', normal_image)
            # cv2.imshow('albedo_image', albedo_image)
            cv2.waitKey()
            cv2.destroyAllWindows()
        if do_write:
            #print("shading min: {}, max: {}".format(np.min(shading_image), np.max(shading_image)))
            outdir = "/disk/sdb1/avatars/extra_repo/TEMP"
            if "texture" in what_to_save:
                cv2.imwrite(os.path.join(outdir, FILE_renders_tex.format(frame_id)), textured_image)
            if "albedo" in what_to_save:
                cv2.imwrite(os.path.join(outdir, FILE_albedos.format(frame_id)), albedo_image)
            if "shading" in what_to_save:
                cv2.imwrite(os.path.join(outdir, FILE_shadows.format(frame_id)), shading_image)
            if "pos_mask" in what_to_save:
                cv2.imwrite(os.path.join(outdir, FILE_RENDERS_POS.format(frame_id)), pos_mask)
            if "normals" in what_to_save:
                cv2.imwrite(os.path.join(outdir, FILE_RENDERS_NORM.format(frame_id)), normal_image)
            exit()

    print("Done! {}, {}".format(subset, video))

# ============================================================================================================

device =  "cuda"
# to test the script
do_write = False
do_show = False

target_size = 512 # resulting renders will be target_scale*scale

texture_type = "head_mask" # only_face, head_mask, head_render
fullhead = True

frames_root = "/disk/sdb1/avatars/dataset_EPE_data2/FullEPEDataset/"
renders_out_root = "/disk/sdb1/avatars/dataset_EPE_data2/FullEPEDataset/"

scale = 2

# ---- video file info ----
domain = "real"
subset = ""  #"train"

# you possibly won't change this (DECA input size)
image_crop_size = 224

what_to_save = ("normals",) # ("normals", "pos_mask", "albedo", "texture", "shading")

# ============================================================================================================

datadir = os.path.join(frames_root, subset, domain)
videos = [videodir for videodir in os.listdir(datadir) if os.path.isdir(os.path.join(datadir, videodir))]
Lv = len(videos)
for i, v in enumerate(videos):
    print(f"{i+1}/{Lv}")
    generate_data(subset=subset, video=v)
