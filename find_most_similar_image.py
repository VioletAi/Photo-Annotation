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

def lookat(center, target, up):
    """
    From: LAR-Look-Around-and-Refer
    https://github.com/eslambakr/LAR-Look-Around-and-Refer
    https://github.com/isl-org/Open3D/issues/2338
    https://stackoverflow.com/questions/54897009/look-at-function-returns-a-view-matrix-with-wrong-forward-position-python-im
    https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
    https://www.youtube.com/watch?v=G6skrOtJtbM
    f: forward
    s: right
    u: up
    """
    f = target - center
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    u = u / np.linalg.norm(u)

    m = np.zeros((4, 4))
    m[0, :-1] = -s
    m[1, :-1] = u
    m[2, :-1] = f
    m[-1, -1] = 1.0

    t = np.matmul(-m[:3, :3], center)
    m[:3, 3] = t

    return m


def get_extrinsic(camera_location,target_location):
    camera_location=np.array(camera_location)
    target_location=np.array(target_location)

    up_vector = np.array([0, 0, -1])
    pose_matrix = lookat(camera_location, target_location, up_vector)
    pose_matrix_calibrated = np.transpose(np.linalg.inv(np.transpose(pose_matrix)))
    return pose_matrix_calibrated


def render_mesh(pose, intrin_path, image_width, image_height, mesh, name, device):
    """
    Given the mesh in PyTorch3D format, render images with a defined pose, intrinsic matrix, and image dimensions
    """

    background_color = (1.0, 1.0, 1.0)
    intrinsic_matrix = torch.zeros([4, 4])
    intrinsic_matrix[3, 3] = 1

    intrinsic_matrix_load = intrin_path
    intrinsic_matrix_load_torch = torch.from_numpy(intrinsic_matrix_load)
    intrinsic_matrix[:3, :3] = intrinsic_matrix_load_torch
    extrinsic_load = pose
    camera_to_world = torch.from_numpy(extrinsic_load)
    world_to_camera = torch.inverse(camera_to_world)
    fx, fy, cx, cy = (
        intrinsic_matrix[0, 0],
        intrinsic_matrix[1, 1],
        intrinsic_matrix[0, 2],
        intrinsic_matrix[1, 2],
    )
    width, height = image_width, image_height
    rotation_matrix = world_to_camera[:3, :3].permute(1, 0).unsqueeze(0)
    translation_vector = world_to_camera[:3, 3].reshape(-1, 1).permute(1, 0)
  # print("translation vector",translation_vector)
    focal_length = -torch.tensor([[fx, fy]])
    principal_point = torch.tensor([[cx, cy]])
    camera = PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        R=rotation_matrix,
        T=translation_vector,
        image_size=torch.tensor([[height, width]]),
        in_ndc=False,
        device=device,
    )
    lights = AmbientLights(device=device)
    raster_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=HardPhongShader(
            blend_params=BlendParams(background_color=background_color),
            device=lights.device,
            cameras=camera,
            lights=lights,
        ),
    )
    rendered_image = renderer(mesh)
    # print("rendered image shape",rendered_image.shape)
    rendered_image = rendered_image[0].cpu().numpy()

    color = rendered_image[..., :3]
    # print("color: ",color[0])
    color_image = Image.fromarray((color * 255).astype(np.uint8))
    # display(color_image)
    # color_image.save(name)
    return color_image,camera