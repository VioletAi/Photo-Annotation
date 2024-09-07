import numpy as np
import torch
from pytorch3d.renderer import (
    PerspectiveCameras)

def construct_bbox_corners(center, box_size):
    """ Calculate the 8 corners of the 3D bounding box """
    sx, sy, sz = box_size
    x_corners = [sx / 2, sx / 2, -sx / 2, -sx / 2, sx / 2, sx / 2, -sx / 2, -sx / 2]
    y_corners = [sy / 2, -sy / 2, -sy / 2, sy / 2, sy / 2, -sy / 2, -sy / 2, sy / 2]
    z_corners = [sz / 2, sz / 2, sz / 2, sz / 2, -sz / 2, -sz / 2, -sz / 2, -sz / 2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)

    return corners_3d

def calculate_iou(box1, box2):
    """ Calculate Intersection over Union (IoU) between two bounding boxes """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def get_2d_bounding_box(corners_2d):
    """ Get 2D bounding box from 2D corners """
    x_min, y_min = torch.min(corners_2d, dim=0)[0]
    x_max, y_max = torch.max(corners_2d, dim=0)[0]
    return [x_min.item(), y_min.item(), x_max.item(), y_max.item()]

def check_visibility(objects, camera, image_width, image_height):
    points_3d = objects[:, :3]
    center_in_camera_coordinate = camera.get_world_to_view_transform().transform_points(points_3d)
    bbox_sizes = objects[:, 3:]

    # Calculate depths of objects from the camera
    depths_in_camera_view = center_in_camera_coordinate[:, 2]
    # print("Depths:", depths_in_camera_view)

    # Sort objects based on their depth (closest first)
    sorted_indices = torch.argsort(depths_in_camera_view)
    # print("Sorted indices based on depth:", sorted_indices)
    visible_indices = []

    # Loop through each object and check for occlusion and image bounds
    for i in sorted_indices:
        center = points_3d[i].cpu().numpy()
        size = bbox_sizes[i].cpu().numpy()

        # Calculate 3D corners and transform to 2D image coordinates
        corners_3d = construct_bbox_corners(center, size)
        corners_3d = torch.tensor(corners_3d, dtype=torch.float32, device=points_3d.device)

        corners_2d = camera.transform_points_screen(corners_3d)[:,:2]
        # Check if any corner is within image bounds
        if torch.any((corners_2d[:, 0] >= 0) & (corners_2d[:, 0] < image_width) &
                     (corners_2d[:, 1] >= 0) & (corners_2d[:, 1] < image_height)):
            bbox_2d = get_2d_bounding_box(corners_2d)

            # Check IoU with previously visible objects
            occluded = False
            for vi in visible_indices:
                vi_center = points_3d[vi].cpu().numpy()
                vi_size = bbox_sizes[vi].cpu().numpy()
                vi_corners_3d = construct_bbox_corners(vi_center, vi_size)
                vi_corners_3d = torch.tensor(vi_corners_3d, dtype=torch.float32, device=points_3d.device)
                vi_corners_2d = camera.transform_points_screen(vi_corners_3d)[:,:2]
                vi_bbox_2d = get_2d_bounding_box(vi_corners_2d)
                iou = calculate_iou(bbox_2d, vi_bbox_2d)
                if iou > 0.2:
                    # print(f"Object {i} is occluded by object {vi} with IoU {iou}.")
                    occluded = True
                    break

            if not occluded:
                visible_indices.append(i.item())
    
    return visible_indices