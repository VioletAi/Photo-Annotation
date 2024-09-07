import sys
import json
import numpy as np
import torch
from pytorch3d.io import IO
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    AmbientLights,
    HardPhongShader,
    BlendParams)

from PIL import Image, ImageDraw, ImageFont

import numpy as np

from compute_not_occluded_idx import *
from find_most_similar_picture import *

import os
sys.path.append(os.path.join(os.getcwd()))

SCANNET_ROOT = "/home/wa285/rds/hpc-work/Thesis/dataset/scannet/raw/scans"
SCANNET_MESH = os.path.join(SCANNET_ROOT, "{}/{}_vh_clean_2.ply")  # scene_id, scene_id
SCANNET_META = os.path.join(SCANNET_ROOT, "{}/{}.txt")  # scene_id, scene_id