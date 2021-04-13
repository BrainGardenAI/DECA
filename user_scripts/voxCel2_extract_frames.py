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
frame_data = pd.read_csv(annopath, skiprows=6, header=0, skip_blank_lines=False, sep='\t')
frame_data.info()
print(frame_data.columns)

# read video and extract target frame_data
cap = cv2.VideoCapture(os.path.join(out_video_dir, os.path.splitext(part)[0])+'.mp4')
# 25 fps
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
assert fps == 25, "FPS != 25"
# cap.set(cv2.CAP_PROP_FPS, 25)

for idx, row in frame_data.iterrows():
    fn = int(row['FRAME '])
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frame = cap.read()
    H, W, _ = frame.shape
    x = int(round(row['X ']*W))
    y = int(round(row['Y ']*H))
    w = int(round(row['W ']*W))
    h = int(round(row['H ']*H))
    cv2.rectangle(frame, pt1=(x,y), pt2=(x+w,y+h), color=(0, 255, 0), thickness=2)
    cv2.imshow('Frame {}'.format(fn), frame)
    cv2.waitKey()
    cv2.destroyAllWindows()
cap.release()

print("Done!")
