# -*- coding: utf-8 -*-
# conda env aphrodite: lg-avatar-pytorch3d (/home/developer/miniconda3/envs/lg-avatar-pytorch3d)

import os
import sys
import shutil
import argparse
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.datasets.constants import *
from decalib.utils.config import cfg as deca_cfg

def main(args):
    output_root = args.output_root
    device = args.device
    os.makedirs(output_root, exist_ok=True)

    save_video_csv = None
    save_missing_bbox_csv = None
    if args.save_video_csv:
        save_video_csv = os.path.join(args.vidset_root, "videos_df.csv")
    if args.save_missing_bbox_csv:
        missing_bbox = []
        save_missing_bbox_csv = os.path.join(args.vidset_root, "missing_bbox.csv")

    # load test images 
    videodata = datasets.VideoDataset(vidset_path=args.vidset_root,
                                      frame_dataset_root=args.output_root,
                                      save_df_path=save_video_csv,
                                      extract_frames=args.extract_frames)
    print(len(videodata))

    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca = DECA(config = deca_cfg, device=device)

    # loop over videos
    for i in tqdm(range(len(videodata))):
        # get frames for each video
        if args.extract_frames:
            video_df, imagepath_list, video_anno_dir = videodata[i]
        else:
            video_df, frameset_root = videodata[i]
            imagepath_list = [os.path.join(frameset_root, item) for item in os.listdir(frameset_root) if os.path.splitext(item)[1] in FRAME_SUFFIX]
            video_anno_dir = os.path.split(frameset_root)[0]

        framedata = datasets.TestData(testpath=imagepath_list, iscrop=args.iscrop, face_detector=args.detector)
        out_dir = os.path.join(video_anno_dir, DIR_PARAMS)
        if args.replace_params:
            shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, exist_ok=False)
        # loop over frames
        for i in tqdm(range(len(framedata)), leave=False):
            name = framedata[i]['imagename']
            isFace = framedata[i]['isFace']
            # if head bbox was found, save parameters
            if isFace:
                images = framedata[i]['image'].to(device)[None,...]
                codedict = deca.encode(images)
                codedict["bbox"] = np.array(framedata[i]['bbox'], dtype=np.float)
                # save .npy param file
                paramfile_name = os.path.join(out_dir, FILE_PARAMS_TEMP.format(os.path.splitext(name)[0]))
                out_array = np.zeros(shape=PARAM_DICT_ARR_SHAPE)
                for key in PARAM_DICT_ARR:
                    start, stop, _, _ = PARAM_DICT_ARR[key]
                    out_array[start:stop] = codedict[key].cpu().numpy().flatten() if key != "bbox" else codedict[key]
                np.save(paramfile_name, out_array)

            # If head bbox wasn't found and args.save_missing_bbox_csv == True, save frame to the file for missed frames
            elif args.save_missing_bbox_csv:
                missing_bbox.append([name, framedata[i]['imagepath'], video_df["actor"], video_df["subset"],
                                     os.path.split(video_df["filename"])[1]])

    if args.save_missing_bbox_csv:
        missing_bbox = pd.DataFrame(missing_bbox, columns=COLUMNS_BBOX_MISSING)
        missing_bbox.to_csv(save_missing_bbox_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--vidset_root', default='/disk/sdb1/avatars/video_data', type=str,
                        help='path to the dataset with videos')
    parser.add_argument('-s', '--output_root', default='/disk/sdb1/avatars/dataset_processed', type=str,
                        help='path to the processed dataset that contains frames')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu, cuda for using gpu (default option)')
    # process test images
    parser.add_argument('--extract_frames', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract and save frames from video or just look for them in output_root folder')
    parser.add_argument('--replace_params', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to replace params')
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details' )
    # save
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--save_video_csv', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save video dataset as .csv file, as vidset_root/videos_df.csv')
    parser.add_argument('--save_missing_bbox_csv', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save information (as a .csv file) about frames for which head bboxes weren\'t detected')
    main(parser.parse_args())