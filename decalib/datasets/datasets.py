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
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
import scipy.io

from . import detectors
from decalib.datasets.constants import *

def video2sequence(inp_video_path, out_framedir=None):
    if out_framedir is None: out_framedir = os.path.split(inp_video_path)[0] #video_path.rsplit('.')[0]
    os.makedirs(out_framedir, exist_ok=True)
    vidcap = cv2.VideoCapture(inp_video_path)
    success,image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        imagepath = os.path.join(out_framedir, "{:04d}.jpg".format(count))
        cv2.imwrite(imagepath, image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
        imagepath_list.append(imagepath)
    print('video frames are stored in {}'.format(out_framedir))
    return imagepath_list

def bbox2point(left, right, top, bottom, type='bbox'):
    ''' bbox from detector and landmarks are different
    '''
    # TODO: make it more robust and without these magic numbers
    if type=='kpt68':
        old_size = (right - left + bottom - top)/2*1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
    elif type=='bbox':
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
    elif type=='no_detection':
        old_size = 0
        center = np.array([0, 0])
    else:
        raise NotImplementedError
    return old_size, center

class TestData(Dataset):
    def __init__(self, testpath, iscrop=True, crop_size=224, scale=1.25, face_detector='mtcnn', old_transform=True):
        '''
            testpath: folder, imagepath_list, image path, video path
        '''
        if isinstance(testpath, list):
            self.imagepath_list = testpath
        elif os.path.isdir(testpath): 
            self.imagepath_list = glob(testpath + '/*.jpg') +  glob(testpath + '/*.png') + glob(testpath + '/*.bmp')
        elif os.path.isfile(testpath) and (testpath[-3:] in ['jpg', 'png', 'bmp']):
            self.imagepath_list = [testpath]
        elif os.path.isfile(testpath) and (testpath[-3:] in ['mp4', 'csv', 'vid', 'ebm']):
            self.imagepath_list = video2sequence(testpath)
        else:
            print(f'please check the test path: {testpath}')
            exit()
        print('total {} images'.format(len(self.imagepath_list)))
        self.imagepath_list = sorted(self.imagepath_list)
        # crop square with a head, this is side length of this square (before scaling)
        self.crop_size = crop_size
        # bbox side scale (crop not exactly bbox, but a square with a side of bbox_size*scale)
        self.scale = scale
        self.iscrop = iscrop
        self.old_transform = old_transform
        self.resolution_inp = crop_size
        if face_detector == 'fan':
            self.face_detector = detectors.FAN()
        # elif face_detector == 'mtcnn':
        #     self.face_detector = detectors.MTCNN()
        else:
            print(f'please check the detector: {face_detector}')
            exit()

    def __len__(self):
        return len(self.imagepath_list)

    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]
        imagename = imagepath.split('/')[-1].split('.')[0]

        image = np.array(imread(imagepath))
        if len(image.shape) == 2:
            image = image[:,:,None].repeat(1,1,3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:,:,:3]

        # cv2.imshow('before transform', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        isFace = False
        h, w, _ = image.shape
        if self.iscrop:
            # form paths for possible .mat or .txt file with head bbox
            kpt_matpath = os.path.splitext(imagepath)[0] + '.mat'
            kpt_txtpath = os.path.splitext(imagepath)[0] + '.txt'
            # if there is .mat file with coodinates of head bbox
            if os.path.exists(kpt_matpath):
                kpt = scipy.io.loadmat(kpt_matpath)['pt3d_68'].T
                left = np.min(kpt[:,0])
                right = np.max(kpt[:, 0])
                top = np.min(kpt[:,1])
                bottom = np.max(kpt[:, 1])
                old_size, center = bbox2point(left, right, top, bottom, type='kpt68')
                isFace = True
            # if there is .txt file with coodinates of head bbox
            elif os.path.exists(kpt_txtpath):
                kpt = np.loadtxt(kpt_txtpath)
                left = np.min(kpt[:,0])
                right = np.max(kpt[:, 0])
                top = np.min(kpt[:,1])
                bottom = np.max(kpt[:, 1])
                old_size, center = bbox2point(left, right, top, bottom, type='kpt68')
                isFace = True
            # if there is no file with head bbox but we want it anyway
            else:
                bbox, bbox_type = self.face_detector.run(image)
                # if detector doesn't find head bbox, lie that the whole image is bbox (>_>)
                if len(bbox) < 4:
                    #print('no face detected! run original image') # TODO: надо что-то делать, как-то сохранять эту информацию
                    left = 0
                    top = 0
                    right = h-1
                    bottom = w-1
                # if detector is a good boy and found the face we wanted
                else:
                    left = bbox[0]
                    top = bbox[1]
                    right = bbox[2]
                    bottom = bbox[3]
                    isFace = True
                old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
            size = int(old_size*self.scale)
            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        # if don't even try to do crop
        else:
            src_pts = np.array([[0, 0], [0, h-1], [w-1, 0]])

        DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])

        image = image / 255.
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))

        """
        # our own way to get square crop
        else:
            if self.iscrop:
                assert w >= h
                cut_left = (w - h) // 2
                dst_image = image.copy()[:, cut_left:cut_left+h, :]
                dst_image = cv2.resize(dst_image, (self.resolution_inp, self.resolution_inp))
                dst_image = dst_image / 255.
        """

        # cv2.imshow('after transform', cv2.cvtColor(np.rint(dst_image*255.).astype(np.uint8), cv2.COLOR_RGB2BGR))
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        bbox = (center[0], center[1], size, True) if isFace else (-1, -1, -1, False)
        dst_image = dst_image.transpose(2,0,1)
        return {'image': torch.tensor(dst_image).float(),
                'imagename': imagename,
                'imagepath': imagepath,
                'bbox': bbox,
                'isFace': isFace
                # 'tform': tform,
                # 'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                }

class VideoDataset(Dataset):
    SUBDIRS = ("real", "virtual")
    VIDSUFFIX = (".mp4", "csv", "vid", "ebm")    # TODO: check what types we can use
    COLUMNS = ("actor", "subset", "filename", "filepath")

    def __init__(self, vidset_path, frame_dataset_root, save_df_path=None, extract_frames=False):
        # Make sure that video dataset has required structure
        if not os.path.isdir(vidset_path):
            print("VideoCropDataset in decalib/datasets/datasets.py: argument vidset_path is not a directory!")
            exit()
        self.vidset_path = vidset_path
        self.frame_dataset_root = frame_dataset_root
        self.extract_frames = extract_frames

        # make pandas DataFrame with videos
        self.videos_df = []
        for actordir in os.listdir(vidset_path):
            if not os.path.isdir(os.path.join(vidset_path, actordir)):
                continue
            subactordirs = os.listdir(os.path.join(vidset_path, actordir))
            for subdir in subactordirs:
                if subdir not in self.SUBDIRS:
                    continue
                vidfiles = [[actordir, subdir, vidfile, os.path.join(actordir, subdir, vidfile)] for vidfile in os.listdir(os.path.join(vidset_path, actordir, subdir)) if vidfile.endswith(self.VIDSUFFIX)]
                self.videos_df.extend(vidfiles)
        self.videos_df = pd.DataFrame(self.videos_df, columns=self.COLUMNS)
        if save_df_path:
            self.videos_df.to_csv(save_df_path)

    def __len__(self):
        return len(self.videos_df)

    def __getitem__(self, index):
        video_df = self.videos_df.iloc[index]
        # define output dir for frames of this video
        frameset_root = os.path.join(self.frame_dataset_root, video_df["actor"], video_df["subset"],
                                     os.path.splitext(video_df["filename"])[0], "frames")

        if self.extract_frames:
            # read video and get its fps and number of frames
            video = cv2.VideoCapture(os.path.join(self.vidset_path, video_df["filepath"]))
            fps = video.get(cv2.CAP_PROP_FPS)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            # make output dir for frames of this video
            os.makedirs(frameset_root, exist_ok=False)

            # extract frames and save them
            imagepath_list = []
            count = 0
            success = True
            while success:
                imagepath = os.path.join(frameset_root, "{}.jpg".format(str(count).zfill(max(5, len(str(total_frames))))))
                success, image = video.read()
                if success:
                    cv2.imwrite(imagepath, image)  # save frame as JPEG file
                    imagepath_list.append(imagepath)
                    count += 1
            video.release()

            imagepath_list = sorted(imagepath_list)
            return video_df, imagepath_list, os.path.split(frameset_root)[0]
        else:
            return video_df, frameset_root

class ProcessedDataset(Dataset):
    SUBDIRS_ACTOR = ("real", "virtual")
    FRAME_DIR = "frames"
    PARAM_DIR = "params"
    COLUMNS_FRAMES = ("frame_name", "actor", "subset", "file_path", "param_path", "orig_tex_rndr_path", "orig_norm_rndr_path")
    COLUMNS_FRAMEDIR = ("frame_dir", "actor", "subset", "video_file", "param_dir")

    def __init__(self, frameset_root):
        # Make sure that video dataset has required structure
        if not os.path.isdir(frameset_root):
            print("ProcessedDataset in decalib/datasets/datasets.py: argument frame_dataset_root is not a directory!")
            exit()
        self.frameset_root = frameset_root

        # make pandas DataFrame with frame directories
        self.framedir_df = []
        for actordir in os.listdir(self.frameset_root):
            if not os.path.isdir(os.path.join(self.frameset_root, actordir)):
                continue
            subdirs = os.listdir(os.path.join(self.frameset_root, actordir))
            for subdir in subdirs:
                if subdir not in self.SUBDIRS_ACTOR:
                    continue
                subdir_full = os.path.join(self.frameset_root, actordir, subdir)
                for videodir in os.listdir(subdir_full):
                    if not os.path.isdir(os.path.join(subdir_full, videodir)):
                        continue
                    videodir_full = os.path.join(subdir_full, videodir)
                    videoresults = os.listdir(videodir_full)
                    if self.FRAME_DIR in videoresults:
                        self.framedir_df.append([os.path.join(videodir_full, self.FRAME_DIR),
                                                 actordir,
                                                 subdir,
                                                 videodir,
                                                 os.path.join(subdir_full, videodir, self.PARAM_DIR) if self.PARAM_DIR in videoresults else None])

        self.framedir_df = pd.DataFrame(self.framedir_df, columns=self.COLUMNS_FRAMEDIR)

    def __len__(self):
        return len(self.framedir_df)

    def __getitem__(self, index):
        framedir_df = self.framedir_df.iloc[index]
        frame_dir = framedir_df["frame_dir"]
        param_dir = framedir_df["param_dir"]
        frames = os.listdir(frame_dir)

        frames_params = []
        for frame_name in frames:
            if not frame_name.endswith(FRAME_SUFFIX) or param_dir is None:
                continue
            frame_path = os.path.join(frame_dir, frame_name)
            param_path = os.path.join(param_dir, FILE_PARAMS_TEMP.format(os.path.splitext(frame_name)[0]))
            if not os.path.isfile(param_path):
                continue
            frames_params.append([frame_path, param_path])

        return frames_params

class CombinedDataset(Dataset):

    def __init__(self, frameset_dir, device='cuda'):
        # Make sure that video dataset has required structure
        if not os.path.isdir(frameset_dir):
            print("ProcessedDataset in decalib/datasets/datasets.py: argument frame_dataset_dir is not a directory!")
            exit()

        self.device = device
        self.frameset_dir = frameset_dir
        frames_source_dir = os.path.join(frameset_dir, DIR_SOURCE, DIR_FRAMES)
        frames_target_dir = os.path.join(frameset_dir, DIR_TARGET, DIR_FRAMES)
        params_source_dir = os.path.join(frameset_dir, DIR_SOURCE, DIR_PARAMS)
        params_target_dir = os.path.join(frameset_dir, DIR_TARGET, DIR_PARAMS)
        self.frames_source = sorted(os.listdir(frames_source_dir))
        self.frames_target = sorted(os.listdir(frames_target_dir))
        self.params_source = sorted(os.listdir(params_source_dir))
        self.params_target = sorted(os.listdir(params_target_dir))
        assert len(self.frames_source) == len(self.frames_target) == len(self.params_source) == len(self.params_target), \
            "We must have the same number of frames and param files for source and target!"

        self.L = len(self.params_source)

    def __len__(self):
        return self.L

    def __getitem__(self, index):
        source_frame_path = os.path.join(self.frameset_dir, DIR_SOURCE, DIR_FRAMES, self.frames_source[index])
        target_frame_path = os.path.join(self.frameset_dir, DIR_TARGET, DIR_FRAMES, self.frames_target[index])
        source_params_path = os.path.join(self.frameset_dir, DIR_SOURCE, DIR_PARAMS, self.params_source[index])
        target_params_path = os.path.join(self.frameset_dir, DIR_TARGET, DIR_PARAMS, self.params_target[index])

        in_array = np.load(source_params_path)
        source_codedict = {}
        for key in PARAM_DICT_ARR:
            start, stop, newshape, paramtype = PARAM_DICT_ARR[key]
            arr = np.reshape(in_array[start:stop], newshape=newshape)
            source_codedict[key] = torch.tensor(arr).type(paramtype).to(self.device) if key != "bbox" else arr

        in_array = np.load(target_params_path)
        target_codedict = {}
        for key in PARAM_DICT_ARR:
            start, stop, newshape, paramtype = PARAM_DICT_ARR[key]
            arr = np.reshape(in_array[start:stop], newshape=newshape)
            target_codedict[key] = torch.tensor(arr).type(paramtype).to(self.device) if key != "bbox" else arr

        # combine parameters
        combined_codedict = target_codedict
        #combined_codedict["pose"] = source_codedict["pose"]
        combined_codedict["exp"] = source_codedict["exp"]
        combined_codedict["tex"] = source_codedict["tex"]
        combined_codedict["cam"] = source_codedict["cam"]

        return source_codedict, target_codedict, target_frame_path, source_frame_path