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

# for testing the script
do_write = False
do_show = False

def main(args):
    target_size = 224
    frameset_root = args.frame_dataset_root
    device = args.device

    # make a list of all frame directories that we'll process
    framedir_data = datasets.ProcessedDataset(frameset_root)

    # decide what texture renders and meshes we are saving now, and set corresponding DECA config parameters
    if args.texture_type == "head_mask":
        DIR_meshes = DIR_MESHES_HEADMASK
        DIR_renders_tex = DIR_RENDERS_TEX_HEADMASK
        FILE_renders_tex = FILE_RENDERS_TEX_HEADMASK
        FILE_meshes_temp = FILE_MESHES_COARSE_TEMP_HEADMASK
        deca_cfg.model.use_tex = False
        deca_cfg.model.no_flametex_model = True # False is ok too, it doesn't matter here
    elif args.texture_type == "head_render":
        DIR_meshes = DIR_MESHES_HEADREND
        DIR_renders_tex = DIR_RENDERS_TEX_HEADREND
        FILE_renders_tex = FILE_RENDERS_TEX_HEADREND
        FILE_meshes_temp = FILE_MESHES_COARSE_TEMP_HEADREND
        deca_cfg.model.use_tex = True
        deca_cfg.model.no_flametex_model = False
    else:
        DIR_meshes = DIR_MESHES
        DIR_renders_tex = DIR_RENDERS_TEX
        FILE_renders_tex = FILE_RENDERS_TEX
        FILE_meshes_temp = FILE_MESHES_COARSE_TEMP
        deca_cfg.model.use_tex = True
        deca_cfg.model.no_flametex_model = True

    # run DECA
    deca = DECA(config=deca_cfg, device=device)

    # loop over directories with frames extracted from video
    for i in tqdm(range(len(framedir_data))):
        videoinfo = framedir_data.framedir_df.iloc[i]

        for j in tqdm(range(len(framedir_data[i])), leave=False, desc="[{}] {} [{}..{}]:".format(videoinfo["actor"],
                                                                                        videoinfo["subset"][:4],
                                                                                        videoinfo["video_file"][:10],
                                                                                        videoinfo["video_file"][-6:])):
            frame_path, param_path = framedir_data[i][j]
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
                continue

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
            opdict["light"] = None #codedict["light"] # using it results in too dark renders
            # first save mesh in .obj file
            if args.saveMeshes:
                frame_dir, frame_name = os.path.split(frame_path)
                frame_id = os.path.splitext(frame_name)[0]
                dir, frame_dir = os.path.split(frame_dir)
                assert frame_dir == DIR_FRAMES, "Video frames aren't placed in the .../frames directory!"
                mesh_path = os.path.join(dir, DIR_meshes, FILE_meshes_temp.format(frame_id))
                os.makedirs(os.path.join(dir, DIR_meshes), exist_ok=True)
            else:
                temp_dir = os.path.join(os.path.abspath(os.getcwd()), "temp")
                os.makedirs(temp_dir, exist_ok=False)
                mesh_path = os.path.join(temp_dir, "temp_mesh.obj")
            # save meshes (temporarily or permanently) and texture
            deca.save_obj_my_format(filename=mesh_path, opdict=opdict, albedo_to_opdict=True)

            # then get renders
            results = deca.get_renderings(target_size=target_size, mesh_file=mesh_path, opdict=opdict)
            textured_image, normal_image, albedo_image = results
            if not args.saveMeshes:
                shutil.rmtree(temp_dir, ignore_errors=True)

            # save renders
            os.makedirs(os.path.join(dir, DIR_RENDERS_ORIG, DIR_renders_tex), exist_ok=True)
            os.makedirs(os.path.join(dir, DIR_RENDERS_ORIG, DIR_RENDERS_NORM), exist_ok=True)
            texture_path = os.path.join(dir, DIR_RENDERS_ORIG, DIR_renders_tex, FILE_renders_tex.format(frame_id))
            normal_path = os.path.join(dir, DIR_RENDERS_ORIG, DIR_RENDERS_NORM, FILE_RENDERS_NORM.format(frame_id))
            cv2.imwrite(texture_path, textured_image)
            cv2.imwrite(normal_path, normal_image)

            if do_show:
                cv2.imshow('frame', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                cv2.imshow('textured', textured_image)
                cv2.imshow('normal', normal_image)
                # cv2.imshow('albedo_image', albedo_image)
                cv2.waitKey()
                cv2.destroyAllWindows()
            if do_write:
                outdir = "/disk/sdb1/avatars/sveta/TEMP/"
                cv2.imwrite(os.path.join(outdir, FILE_renders_tex.format(frame_id)), textured_image)
                #cv2.imwrite(os.path.join(outdir, FILE_RENDERS_NORM.format(frame_id)), normal_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-r', '--frame_dataset_root', default='/disk/sdb1/avatars/dataset_processed', type=str,
                        help='path to the processed dataset that contains frames')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu, cuda for using gpu (default option)')
    parser.add_argument('--texture_type', default="only_face", type=str,
                        help='what type of texture render we want: only_face, head_mask, head_render.')
    parser.add_argument('--saveMeshes', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save .obj files.')
    main(parser.parse_args())