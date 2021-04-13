import os
import cv2
import pandas as pd
from pytube import YouTube

YT_BASE = "https://www.youtube.com/watch?v={}"

vox2anno_root = "/home/s.orlova@dev.braingarden.ai/Data/vox2"
setdir = "txt"
id = "id00015"
reference = "0iQWqFw6FOU"
output_dir = "videos"

parts = os.listdir(os.path.join(vox2anno_root, setdir, id, reference))

# pick a file
part = parts[0]
annopath = os.path.join(vox2anno_root, setdir, id, reference, part)
out_video_dir = os.path.join(vox2anno_root, output_dir, id, reference)

# parse heading
heading = None
with open(annopath) as f:
    heading = dict(tuple(f.readline().replace(' ', '').replace('\t', '').strip().split(":")) for _ in range(5))

# read data
frames = pd.read_csv(annopath, skiprows=6, header=0, skip_blank_lines=False, sep='\t')
frames.info()

# download video
yt = YouTube(YT_BASE.format(heading["Reference"]))
strms = yt.streams
#yt = yt.streams.get_highest_resolution()
os.makedirs(out_video_dir, exist_ok=True)
yt.streams.get_by_itag(135).download(output_path=out_video_dir, filename=os.path.splitext(part)[0])

# read video and extract target frames
cap = cv2.VideoCapture(os.path.join(out_video_dir, os.path.splitext(part)[0]))
# 25 fps
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = cap.get(7)
cap.set(cv2.CAP_PROP_FPS, 25)

