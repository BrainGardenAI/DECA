# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import cv2
import json
import torch
import argparse
import numpy as np
from time import time
from tqdm import tqdm
from scipy.io import savemat
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg, get_sex, set_sex

def main(args):
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # load test images 
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector, old_transform=args.old_transform, scale=args.crop_scale)

    # run DECA
    deca_cfg.model.use_tex = args.useTex
    set_sex(deca_cfg, sex='m')
    deca = DECA(config = deca_cfg, device=device)
    # for i in range(len(testdata)):
    for i in tqdm(range(len(testdata))):
        name = testdata[i]['imagename']
        bbox = testdata[i]['bbox']
        images = testdata[i]['image'].to(device)[None,...]
        codedict = deca.encode(images)
        codedict["bbox"] = np.array(bbox, dtype=np.float)

        # save as npy
        npy_name = os.path.join(savefolder, 'params_' + name + '.npy')
        dict_to_array = {"shape": (0, 100, (1,100), torch.float32),
                         "exp": (100, 150, (1,50), torch.float32),
                         "tex": (150, 200, (1,50), torch.float32),
                         "pose": (200, 206, (1,6), torch.float32),
                         "cam": (206, 209, (1,3), torch.float32),
                         "light": (209, 236, (1,9,3), torch.float32),
                         "detail": (236, 364, (1,128), torch.float32),
                         "bbox": (364, None, (4,), np.float)}
        out_array = np.zeros(shape=368)
        for key in dict_to_array:
            start, stop, _, _ = dict_to_array[key]
            out_array[start:stop] = codedict[key].cpu().numpy().flatten() if key != "bbox" else codedict[key]
        np.save(npy_name, out_array)

        del bbox

        in_array = np.load(npy_name)
        codedict2 = {}
        for key in dict_to_array:
            start, stop, newshape, paramtype = dict_to_array[key]
            arr = np.reshape(in_array[start:stop], newshape=newshape)
            codedict2[key] = torch.tensor(arr).type(paramtype).to(device) if key != "bbox" else arr

        # read image
        imagepath = testdata[i]['imagepath']
        image = np.array(imread(imagepath))
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(1, 1, 3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]
        h, w, _ = image.shape
        # crop head with saved annotation
        cx, cy, size, isFace = codedict2["bbox"].tolist()
        if isFace:
            target_size = 224
            src_pts = np.array([[cx-size/2, cy-size/2], [cx - size/2, cy+size/2], [cx+size/2, cy-size/2]])
            DST_PTS = np.array([[0, 0], [0, target_size - 1], [target_size - 1, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)
            image = image / 255.
            dst_image = warp(image, tform.inverse, output_shape=(target_size, target_size))
            dst_image = dst_image.transpose(2, 0, 1)

        codedict2["images"] = torch.tensor(dst_image).float().to(device)[None,...]
        codedict = codedict2

        opdict, visdict = deca.decode(codedict) #tensor
        if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages:
            os.makedirs(os.path.join(savefolder, name), exist_ok=True)
        # -- save results
        if args.saveDepth:
            depth_image = deca.render.render_depth(opdict['transformed_vertices']).repeat(1,3,1,1)
            visdict['depth_images'] = depth_image
            cv2.imwrite(os.path.join(savefolder, name, name + '_depth.jpg'), util.tensor2image(depth_image[0]))
        if args.saveKpt:
            np.savetxt(os.path.join(savefolder, name, name + '_kpt2d.txt'), opdict['landmarks2d'][0].cpu().numpy())
            np.savetxt(os.path.join(savefolder, name, name + '_kpt3d.txt'), opdict['landmarks3d'][0].cpu().numpy())
        if args.saveObj:
            deca.save_obj(os.path.join(savefolder, name, name + '.obj'), opdict)
        if args.saveMat:
            opdict = util.dict_tensor2npy(opdict)
            savemat(os.path.join(savefolder, name, name + '.mat'), opdict)
        if args.saveVis:
            cv2.imwrite(os.path.join(savefolder, name + '_vis.jpg'), deca.visualize(visdict))
        if args.saveImages:
            for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images']:
                if vis_name not in visdict.keys():
                    continue
                image = util.tensor2image(visdict[vis_name][0])
                cv2.imwrite(os.path.join(savefolder, name, name + '_' + vis_name +'.jpg'), util.tensor2image(visdict[vis_name][0]))
    print(f'-- please check the results in {savefolder}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='TestSamples/examples/results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--old_transform', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use old (original) transformation method')
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details' )
    parser.add_argument('--crop_scale', default=1.25, type=float,
                        help='scale for extending face bbox')
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints' )
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image' )
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                            Note that saving objs could be slow' )
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat' )
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    main(parser.parse_args())