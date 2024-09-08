import sys
import numpy as np
import torch
from pytorch3d.io import IO
import json
import cv2

from PIL import Image, ImageDraw, ImageFont
from take_photo import *
from skimage.metrics import structural_similarity as ssim


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
    np_imageA = np.array(image1)
    np_imageB = np.array(image2)
    
    # Convert images to grayscale
    grayA = cv2.cvtColor(np_imageA, cv2.COLOR_RGB2GRAY)
    grayB = cv2.cvtColor(np_imageB, cv2.COLOR_RGB2GRAY)
    
    # Compute SSIM between the two images
    score, _ = ssim(grayA, grayB, full=True)
    return score

def compute_mean_euclidean_distance_3D(points_A, points_B):
    """
    Computes the mean Euclidean distance between two sets of 3D points using PyTorch tensors.
    points_A: (N, 3) PyTorch tensor of 3D points in camera coordinates for target pose
    points_B: (N, 3) PyTorch tensor of 3D points in camera coordinates for a candidate pose
    
    Returns:
    Mean Euclidean distance between the two sets of points.
    """
    # Compute the Euclidean distance between corresponding points in A and B
    distances = torch.norm(points_A - points_B, dim=1)  # Norm along the point dimension (N, 3)
    
    # Return the mean distance
    return torch.mean(distances)

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
    vertices_world = mesh.verts_packed()
    
    target_image ,target_camera =render_mesh(
    view_matrix,
    depth_intrinsic[:3, :3],
    image_width,
    image_height,
    mesh,
    f"./image_rendered_angle.png", device=device)

    projected_coordinate_camera_target= target_camera.get_world_to_view_transform().transform_points(vertices_world)

    # find the most similar matrix
    candidate_poses = f"/home/wa285/rds/hpc-work/Thesis/image_dataset_pose/images_and_pose/{scene_id}/pose"

    best_similarity = float('inf')
    best_extrinsic = None
    best_file_name = None

    for file_name in os.listdir(candidate_poses):
        file_path = os.path.join(candidate_poses, file_name)
        
        # Read the matrix from the current file
        candidate_extrinsic_matrix = read_matrix_from_file(file_path)
        candidate_image ,camera_candidate =render_mesh(
        candidate_extrinsic_matrix,
        depth_intrinsic[:3, :3],
        image_width,
        image_height,
        mesh,
        f"./candidate{file_name}.png", device=device)

        projected_coordinate_camera_candidate= camera_candidate.get_world_to_view_transform().transform_points(vertices_world)

        # Compute similarity using normalized cross-correlation
        similarity = compute_mean_euclidean_distance_3D(projected_coordinate_camera_target, projected_coordinate_camera_candidate)
        
        if similarity < best_similarity:
            best_similarity = similarity
            best_extrinsic = candidate_extrinsic_matrix
            best_file_name = file_name
    
    print(best_file_name)
    return best_extrinsic, best_similarity


if __name__ == "__main__":

    with open("/home/wa285/rds/hpc-work/Thesis/annotated_cameras/scene0011_00.anns.json", 'r') as file:
        parsed_data = json.load(file)
    instance_attribute_file = f"/home/wa285/rds/hpc-work/Thesis/Thesis-Chat-3D-v2/annotations/scannet_mask3d_val_attributes.pt"
    
    test_annotated_camera=parsed_data[13]['camera']

    file_path = '/home/wa285/rds/hpc-work/Thesis/download_intinsic_matrix/output/scene0011_00/intrinsic/intrinsic_color.txt'
    save_path="hello3.png"
    print(parsed_data[13])
    capture_one_image("scene0011_00", test_annotated_camera, instance_attribute_file, 1296, 968, save_path)