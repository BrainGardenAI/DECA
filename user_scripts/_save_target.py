import os
import shutil
from decalib.datasets.constants import *

on_server = False

L = 445
root = "/" if on_server else "/home/s.orlova@dev.braingarden.ai/MOUNTED/aphrodite"
dir = "disk/sdb1/avatars/dataset_processed/NormanReedus/real"
dirs = ["Norman Reedus Answers the Web's Most Searched Questions _ WIRED-z5MIL3KVHxo-00.10.51.317-00.11.07.333-seg34",
        "Norman Reedus Answers the Web's Most Searched Questions _ WIRED-z5MIL3KVHxo-00.13.10.122-00.13.27.973-seg39",
        ]
target_dir = "disk/sdb1/avatars/dataset_combined/2_445/target"

os.makedirs(os.path.join(root, target_dir, DIR_FRAMES), exist_ok=False)
os.makedirs(os.path.join(root, target_dir, DIR_PARAMS), exist_ok=False)

all_frames_paths = []
all_params_paths = []
for d in dirs:
        frame_paths = [os.path.join(root, dir, d, DIR_FRAMES, file) for file in os.listdir(os.path.join(root, dir, d, DIR_FRAMES)) if os.path.splitext(file)[1] in FRAME_SUFFIX]
        frame_numbers = [os.path.splitext(os.path.split(fp)[1])[0] for fp in frame_paths]
        param_paths = [os.path.join(root, dir, d, DIR_PARAMS, FILE_PARAMS_TEMP.format(fnum)) for fnum in frame_numbers]
        all_frames_paths.extend(frame_paths)
        all_params_paths.extend(param_paths)
        if len(all_params_paths) > L:
                break

deb = len(all_params_paths)
assert len(all_params_paths) >= L, "We don't have enough samples!"

# copy files changing their names
for i in range(L):
        frame_from = all_frames_paths[i]
        param_from = all_params_paths[i]
        frame_to = os.path.join(root, target_dir, DIR_FRAMES, FILE_FRAME_TEMP.format(i))
        param_to = os.path.join(root, target_dir, DIR_PARAMS, FILE_PARAMS_TEMP.format(i))
        shutil.copyfile(src=frame_from, dst=frame_to)
        shutil.copyfile(src=param_from, dst=param_to)
        print("{}/{} ready".format(i+1, L))

print("Done!")

