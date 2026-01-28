import numpy as np
import tifffile as tiff
import torch


def organized_pc_to_unorganized_pc(organized_pc):
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])


def read_tiff_organized_pc(path):
    tiff_img = tiff.imread(path)
    return tiff_img


def resize_organized_pc(organized_pc, target_height=224, target_width=224, tensor_out=False, mode="nearest"):
    torch_organized_pc = torch.tensor(organized_pc).permute(2, 0, 1).unsqueeze(dim=0).float()
    torch_resized_organized_pc = torch.nn.functional.interpolate(torch_organized_pc, size=(target_height, target_width),
                                                                 mode=mode)
    if tensor_out:
        return torch_resized_organized_pc.squeeze(dim=0)
    else:
        return torch_resized_organized_pc.squeeze(dim=0).permute(1, 2, 0).numpy()


def organized_pc_to_depth_map(organized_pc):
    return organized_pc[:, :, 2]


def matrix_z(angle):
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([[cos_angle, -sin_angle, 0, 0], [sin_angle, cos_angle, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def matrix_x(angle):
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([[1, 0, 0, 0],
                     [0, cos_angle, -sin_angle, 0],
                     [0, sin_angle, cos_angle, 0],
                     [0, 0, 0, 1]])


def matrix_y(angle):
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([[cos_angle, 0, sin_angle, 0],
                     [0, 1, 0, 0],
                     [-sin_angle, 0, cos_angle, 0],
                     [0, 0, 0, 1]])


def compute_visibility_3d_to_2d(points_3d, correspondence, img_shape=(500, 500)):
    # Initialize visibility array with zeros
    visibility = np.zeros(len(points_3d), dtype=np.int32)

    # Initialize depth map with -inf values
    depth_map = np.full(img_shape, -np.inf)

    max_depth_indices = -np.ones(img_shape, dtype=np.int32)

    # Check bounds and mark out-of-bounds points as invisible
    img_h, img_w = img_shape
    for i, (x, y) in enumerate(correspondence):
        # Check if point is within image bounds
        if 0 <= x < img_w and 0 <= y < img_h:
            z = points_3d[i, 2]
            if z > depth_map[y, x]:
                depth_map[y, x] = z
                max_depth_indices[y, x] = i
        # Points out of bounds are automatically invisible (visibility[i] remains 0)

    for i, (x, y) in enumerate(correspondence):
        # Only check visibility for points within bounds
        if 0 <= x < img_w and 0 <= y < img_h:
            if max_depth_indices[y, x] == i:
                visibility[i] = 1
    return visibility


def back_to_3d(d2_similarity_map, d2_3d_cor, non_zero_index, view_weights=None, ori_resolution=336):
    # _, h, w, _ = d2_similarity_map.shape
    h = torch.sqrt(torch.tensor(d2_3d_cor.shape[2])).int()
    w = h
    b, nv, num_points, _ = d2_3d_cor.shape
    xx = d2_3d_cor[:, :, :, 0].reshape(-1).long()
    yy = d2_3d_cor[:, :, :, 1].reshape(-1).long()
    nbatch = torch.repeat_interleave(torch.arange(0, b * nv)[:, None], num_points).reshape(-1, ).cuda().long()
    d2_similarity_map = d2_similarity_map.permute(0, 3, 1, 2)
    point_logits = d2_similarity_map[nbatch, :, yy, xx]
    point_logits = point_logits.reshape(b, nv, num_points, 2)


    if view_weights is not None:
        vweights = view_weights.reshape(b, nv, 1, 1)
    else:
        vweights = torch.ones((b, nv, 1, 1)).to(point_logits.device)

    is_seen = d2_3d_cor[:, :, :, 2].reshape(b, nv, num_points, 1)
    point_logits = point_logits * vweights * is_seen * non_zero_index


    mask = is_seen.bool() & non_zero_index.bool()  
    masked_vweights = vweights * mask  


    weight_sum = masked_vweights.sum(dim=1, keepdim=True)
    normalized_vweights = torch.where(weight_sum > 0, masked_vweights / weight_sum, masked_vweights)


    weighted_logits = point_logits * normalized_vweights
    point_logits = weighted_logits.sum(1)

    point_logits = point_logits.reshape(b, h, w, 2)
    return point_logits