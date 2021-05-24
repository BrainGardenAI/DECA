import torch
import numpy as np

DIR_FRAMES = "frames"
DIR_PARAMS = "params"
DIR_MESHES = "meshes"
DIR_MESHES_HEADMASK = "meshes_headmask"
DIR_MESHES_HEADREND = "meshes_headrend"
DIR_SOURCE = "source"
DIR_TARGET = "target"
DIR_RENDERS_ORIG = "original_renders"
DIR_RENDERS_TEX = "texture_renders"
DIR_RENDERS_TEX_HEADMASK = "texture_headmask_renders"
DIR_RENDERS_TEX_HEADREND = "texture_headrend_renders"
DIR_RENDERS_ALB = "albedos"
DIR_RENDERS_ALB_HEADMASK = "albedos_headmask"
DIR_RENDERS_ALB_HEADREND = "albedos_headrend"
DIR_RENDERS_SHA = "shading"
DIR_RENDERS_SHA_HEADMASK = "shading_headmask"
DIR_RENDERS_SHA_HEADREND = "shading_headrend"
DIR_RENDERS_POS = "pos_mask"
DIR_RENDERS_NORM = "normal_renders"
FILE_FRAME_TEMP = "{}.jpg"
FILE_PARAMS_TEMP = "flame_{}.npy"
FILE_MESHES_COARSE_TEMP = "mesh_{}.obj"
FILE_MESHES_COARSE_TEMP_HEADMASK = "meshheadmask_{}.obj"
FILE_MESHES_COARSE_TEMP_HEADREND = "meshheadrend_{}.obj"
FILE_MESHES_DETAIL_TEMP = "mesh_{}_detailed.obj"
FILE_RENDERS_TEX = "tex_{}.jpg"
FILE_RENDERS_TEX_HEADMASK = "texheadmask_{}.jpg"
FILE_RENDERS_TEX_HEADREND = "texheadrend_{}.jpg"
FILE_RENDERS_ALB = "alb_{}.jpg"
FILE_RENDERS_ALB_HEADMASK = "albheadmask_{}.jpg"
FILE_RENDERS_ALB_HEADREND = "albheadrend_{}.jpg"
FILE_RENDERS_SHA = "sha_{}.jpg"
FILE_RENDERS_SHA_HEADMASK = "shaheadmask_{}.jpg"
FILE_RENDERS_SHA_HEADREND = "shaheadrend_{}.jpg"
FILE_RENDERS_POS = "pos_{}.jpg"
FILE_RENDERS_NORM = "norm_{}.jpg"
COLUMNS_BBOX = ("cx", "cy", "size", "correct_face")
COLUMNS_BBOX_MISSING = ("frame_name", "frame_path", "actor_dir", "subset", "frame_dir")
FRAME_SUFFIX = (".jpg", ".png", ".jpeg", ".JPG", ".JPEG", ".bmp")
PARAM_EXT = ".npy"

PARAM_DICT_ARR_SHAPE = 368
PARAM_DICT_ARR = {"shape": (0, 100, (1, 100), torch.float32),
                  "exp": (100, 150, (1, 50), torch.float32),
                  "tex": (150, 200, (1, 50), torch.float32),
                  "pose": (200, 206, (1, 6), torch.float32),
                  "cam": (206, 209, (1, 3), torch.float32),
                  "light": (209, 236, (1, 9, 3), torch.float32),
                  "detail": (236, 364, (1, 128), torch.float32),
                  "bbox": (364, None, (4,), np.float)}

