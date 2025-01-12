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

do_show = False

def main(args):
    frameset_dir = args.frame_dataset_dir
    device = args.device

    # make a list of all frame directories that we'll process
    framedir_data = datasets.CombinedDataset(frameset_dir, device=device)

    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.model.no_flametex_model = True
    deca = DECA(config=deca_cfg, device=device)

    # loop over directories with frames extracted from video
    for i in tqdm(range(len(framedir_data))):
        sample = framedir_data[i]

        source_codedict, target_codedict, target_frame_path, source_frame_path = sample

        # read frame (image)
        image = np.array(imread(target_frame_path))
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(1, 1, 3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]
        h, w, _ = image.shape
        # crop head with saved bbox coordinates
        cx, cy, size, isFace = target_codedict["bbox"].tolist()
        if not isFace:
            continue
        target_size = 224
        src_pts = np.array([[cx - size / 2, cy - size / 2], [cx - size / 2, cy + size / 2], [cx + size / 2, cy - size / 2]])
        DST_PTS = np.array([[0, 0], [0, target_size - 1], [target_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        dst_image = image / 255.
        dst_image = warp(dst_image, tform.inverse, output_shape=(target_size, target_size))
        dst_image = dst_image.transpose(2, 0, 1)
        # now we have a dictionary for decoding
        target_codedict["images"] = torch.tensor(dst_image).float().to(device)[None, ...]

        # decode
        opdict_t, visdict_t = deca.decode(target_codedict)

        visdict_t = {x: visdict_t[x] for x in ['inputs', 'shape_detail_images']}
        target_codedict["pose"] = source_codedict["pose"]
        target_codedict["exp"] = source_codedict["exp"]
        target_codedict["cam"] = source_codedict["cam"]
        opdict_transfer, visdict_transfer = deca.decode(target_codedict)
        visdict_t['transferred_shape'] = visdict_transfer['shape_detail_images']
        opdict_transfer['uv_texture_gt'] = opdict_t['uv_texture_gt']

        # we need transferred model
        opdict = opdict_transfer

        # first save mesh in .obj file
        if args.saveMeshes:
            frame_dir, frame_name = os.path.split(target_frame_path)
            frame_id = os.path.splitext(frame_name)[0]
            dir, frame_dir = os.path.split(frame_dir)
            assert frame_dir == DIR_FRAMES, "Video frames aren't placed in the .../frames directory!"
            mesh_path = os.path.join(os.path.split(dir)[0], DIR_MESHES, FILE_MESHES_COARSE_TEMP.format(frame_id))
            os.makedirs(os.path.join(os.path.split(dir)[0], DIR_MESHES), exist_ok=True)
        else:
            temp_dir = os.path.join(os.path.abspath(os.getcwd()), "temp")
            os.makedirs(temp_dir, exist_ok=False)
            mesh_path = os.path.join(temp_dir, "temp_mesh.obj")
        # save meshes (temporarily or permanently) and texture
        deca.save_obj_my_format(filename=mesh_path, opdict=opdict, albedo_to_opdict=True)

        # then get renders
        opdict["light"] = None
        results = deca.get_renderings(target_size=target_size, mesh_file=mesh_path, opdict=opdict)
        textured_image, normal_image, albedo_image = results
        if not args.saveMeshes:
            shutil.rmtree(temp_dir, ignore_errors=True)

        #"""# save renders
        os.makedirs(os.path.join(os.path.split(dir)[0], DIR_RENDERS_ORIG, DIR_RENDERS_TEX), exist_ok=True)
        os.makedirs(os.path.join(os.path.split(dir)[0], DIR_RENDERS_ORIG, DIR_RENDERS_NORM), exist_ok=True)
        texture_path = os.path.join(os.path.split(dir)[0], DIR_RENDERS_ORIG, DIR_RENDERS_TEX, FILE_RENDERS_TEX.format(frame_id))
        normal_path = os.path.join(os.path.split(dir)[0], DIR_RENDERS_ORIG, DIR_RENDERS_NORM, FILE_RENDERS_NORM.format(frame_id))
        cv2.imwrite(texture_path, textured_image)
        cv2.imwrite(normal_path, normal_image)
        #"""

        if do_show:
            source = np.array(imread(source_frame_path))
            hs, w, c = source.shape
            left = int((w-hs)//2)
            source = source[:, left:left+hs, :]
            hi, w, c = image.shape
            left = int((w - hi) // 2)
            image = image[:, left:left+hi, :]
            if hs < hi:
                image = cv2.resize(image, (hs, hs))
            else:
                source = cv2.resize(source, (hi, hi))
            frames = np.hstack((image, source))
            cv2.imshow('source-target', cv2.cvtColor(frames, cv2.COLOR_RGB2BGR))
            cv2.imshow('textured', textured_image)
            cv2.imshow('normal', normal_image)
            cv2.waitKey()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-r', '--frame_dataset_dir', default='/disk/sdb1/avatars/dataset_combined/1_548', type=str,
                        help='path to the processed dataset that contains frames')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu, cuda for using gpu (default option)')
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                                set it to True only if you downloaded texture model')
    parser.add_argument('--saveMeshes', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save .obj files.')
    main(parser.parse_args())