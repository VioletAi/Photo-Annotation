import sys
import numpy as np
import torch
from pytorch3d.io import IO
import json
import cv2

from PIL import Image, ImageDraw, ImageFont
from take_photo import *

import os
sys.path.append(os.path.join(os.getcwd()))

SCANNET_ROOT = "/home/wa285/rds/hpc-work/Thesis/dataset/scannet/raw/scans"
SCANNET_MESH = os.path.join(SCANNET_ROOT, "{}/{}_vh_clean_2.ply")  # scene_id, scene_id
SCANNET_META = os.path.join(SCANNET_ROOT, "{}/{}.txt")  # scene_id, scene_id

def normalized_cross_correlation(image1, image2):
    """
    Computes normalized cross-correlation for color images channel by channel and returns the average.
    Assumes image1 and image2 are PIL images.
    """
    # Convert PIL images to NumPy arrays
    image1_np = np.array(image1)
    image2_np = np.array(image2)
    
    # Initialize cross-correlation scores for each channel (R, G, B)
    ncc_scores = []
    
    for i in range(3):  # Loop over RGB channels
        channel1 = image1_np[:, :, i]
        channel2 = image2_np[:, :, i]
        result = cv2.matchTemplate(channel1, channel2, cv2.TM_CCORR_NORMED)
        ncc_scores.append(result[0][0])  # Extract the cross-correlation value
    
    return np.mean(ncc_scores)  # Return the average similarity across all channels


def read_matrix_from_file(file_path):
    """ Reads a matrix from a text file. """
    return np.loadtxt(file_path)

def capture_one_image(scene_id, annotated_camera, instance_attribute_file, image_width, image_height, save_path):
    device = torch.device("cuda")
    instance_attrs = torch.load(instance_attribute_file)

    mesh_file = f"/home/wa285/rds/hpc-work/Thesis/dataset/scannet/raw/scans/{scene_id}/{scene_id}_vh_clean_2.ply"

    file_path = f'/home/wa285/rds/hpc-work/Thesis/download_intinsic_matrix/output/{scene_id}/intrinsic/intrinsic_color.txt'

    camera_position = annotated_camera["position"]
    lookat_point = annotated_camera["lookat"]

    # get extrinsic matrix
    view_matrix = get_extrinsic(camera_position, lookat_point)
    depth_intrinsic = np.loadtxt(file_path)
    print(view_matrix)
    pt3d_io = IO()
    mesh = pt3d_io.load_mesh(mesh_file, device=device)
    target_image ,camera =render_mesh(
    view_matrix,
    depth_intrinsic[:3, :3],
    image_width,
    image_height,
    mesh,
    f"./image_rendered_angle.png", device=device)
    
    # find the most similar matrix
    candidate_poses = f"/home/wa285/rds/hpc-work/Thesis/image_dataset_pose/images_and_pose/{scene_id}/pose"

    best_similarity = -1
    best_extrinsic = None
    best_file_name = None

    for file_name in os.listdir(candidate_poses):
        file_path = os.path.join(candidate_poses, file_name)
        
        # Read the matrix from the current file
        candidate_extrinsic_matrix = read_matrix_from_file(file_path)
        candidate_image ,camera =render_mesh(
        candidate_extrinsic_matrix,
        depth_intrinsic[:3, :3],
        image_width,
        image_height,
        mesh,
        f"./candidate{file_name}.png", device=device)

        # Compute similarity using normalized cross-correlation
        similarity = normalized_cross_correlation(target_image, candidate_image)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_extrinsic = candidate_extrinsic_matrix
            best_file_name = file_name
    
    print(best_file_name)
    return best_extrinsic, best_similarity


if __name__ == "__main__":

    with open("/home/wa285/rds/hpc-work/Thesis/annotated_cameras/scene0011_00.anns.json", 'r') as file:
        parsed_data = json.load(file)
    instance_attribute_file = f"/home/wa285/rds/hpc-work/Thesis/Thesis-Chat-3D-v2/annotations/scannet_mask3d_val_attributes.pt"
    
    test_annotated_camera=parsed_data[27]['camera']

    file_path = '/home/wa285/rds/hpc-work/Thesis/download_intinsic_matrix/output/scene0011_00/intrinsic/intrinsic_color.txt'
    save_path="hello3.png"
    print(parsed_data[27])
    capture_one_image("scene0011_00", test_annotated_camera, instance_attribute_file, 1296, 968, save_path)