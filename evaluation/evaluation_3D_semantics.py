import torch
import os
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement
import torch.nn.functional as F
import clip
from evaluation.constants import OVOSLAM_COLORED_LABELS, LABEL_TO_COLOR, NYU40, NYU20_INDICES


def read_labels_from_ply(file_path):
    ply_data = PlyData.read(file_path)
    vertex_data = ply_data['vertex'].data
    points = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
    labels = vertex_data['label']  
    return points, labels


def rotate_points(points, angle_deg, axis):
    r = R.from_euler(axis, angle_deg, degrees=True)
    return r.apply(points)

def translate_points(points, translation):
    return points + np.array(translation)



def calculate_metrics(gt, pred):
    total_classes = gt.max().item() + 1
    gt = gt.cpu()
    pred = pred.cpu()

    pred[gt == 0] = 0  # ignore unlabeled in prediction

    intersection = torch.zeros(total_classes)
    union        = torch.zeros(total_classes)
    correct      = torch.zeros(total_classes)
    total        = torch.zeros(total_classes)

    for cls_id in range(1, total_classes):
        intersection[cls_id] = torch.sum((gt == cls_id) & (pred == cls_id)).item()
        union[cls_id]        = torch.sum((gt == cls_id) | (pred == cls_id)).item()
        correct[cls_id]      = torch.sum((gt == cls_id) & (pred == cls_id)).item()
        total[cls_id]        = torch.sum(gt == cls_id).item()

    ious = torch.zeros(total_classes)
    valid_union = union != 0
    ious[valid_union] = intersection[valid_union] / union[valid_union]

    gt_classes = torch.unique(gt)
    gt_classes = gt_classes[gt_classes != 0]
    mean_iou = ious[gt_classes].mean().item() if len(gt_classes) > 0 else 0.0

    valid_mask = (gt != 0)
    correct_predictions = torch.sum((gt == pred) & valid_mask).item()
    total_valid_points  = torch.sum(valid_mask).item()
    accuracy = correct_predictions / total_valid_points if total_valid_points > 0 else 0.0

    class_accuracy = torch.zeros(total_classes)
    non_zero_mask = total != 0
    class_accuracy[non_zero_mask] = correct[non_zero_mask] / total[non_zero_mask]
    mean_class_accuracy = class_accuracy[gt_classes].mean().item() if len(gt_classes) > 0 else 0.0

    return ious, mean_iou, accuracy, mean_class_accuracy

def label_cluster_features(cluster_lang_features, label_features, device="cuda:0"):
    label_features = label_features.to(device)
    cluster_lang_features_tensor = cluster_lang_features.half().to(device)
    label_features = F.normalize(label_features, p=2, dim=1)
    cluster_features_normalized = F.normalize(cluster_lang_features_tensor, p=2, dim=1)

    similarities = cluster_features_normalized @ label_features.t()

    cluster_best_label_ids = torch.argmax(similarities, dim=1)
    return cluster_best_label_ids

def load_text_embeddings(labels):
    clip_model, _ = clip.load("ViT-B/32", device="cuda")
    text_prompts = labels
    text_tokens = clip.tokenize(text_prompts).cuda()
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  
    return text_features

def save_colored_ply(filepath, xyz, rgb):
    if rgb.dtype != np.uint8:
        rgb_255 = (rgb * 255).clip(0, 255).astype(np.uint8)
    else:
        rgb_255 = rgb

    num_verts = xyz.shape[0]

    # Create a structured array suitable for plyfile
    vertex_data = np.zeros(num_verts, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1")
    ])
    vertex_data["x"]    = xyz[:, 0]
    vertex_data["y"]    = xyz[:, 1]
    vertex_data["z"]    = xyz[:, 2]
    vertex_data["red"]  = rgb_255[:, 0]
    vertex_data["green"]= rgb_255[:, 1]
    vertex_data["blue"] = rgb_255[:, 2]

    el = PlyElement.describe(vertex_data, 'vertex')
    PlyData([el], text=True).write(filepath)
    print(f"Saved colored point cloud to: {filepath}")


def evaluate_3D_semantics(xyz, assignments, cluster_lang_feat, dataset_params):
    dataset_dir = dataset_params.source_path
    dataset = dataset_params.type
    scene_name = os.path.basename(dataset_dir)
    output_dir = './output/'
    
    gt_ply = os.path.join(dataset_params.source_path, f"{scene_name}_vh_clean_2.labels.ply")
    points, labels = read_labels_from_ply(gt_ply)
    print(f"[{scene_name}] Loaded {points.shape[0]} vertices. Max label={labels.max()}")

    # target_id_mapping = {}
    # for new_idx, old_label in enumerate(NYU40.keys(), start=1):
    #     target_id_mapping[old_label] = new_idx
    # new_labels = np.zeros_like(labels, dtype=np.int32)
    # for old_value, new_value in target_id_mapping.items():
    #     mask = (labels == old_value)
    #     new_labels[mask] = new_value

    if dataset == "Replica":
        text_labels = []
    elif dataset == "Scannetpp":
        gt_labels = torch.from_numpy(np.array(labels, dtype=np.int32)).long().cuda()
        text_labels = list(NYU40.values())[1:]#[NYU40[i] for i in NYU20_INDICES]

    label_feat = load_text_embeddings(text_labels)
    cluster_to_label = label_cluster_features(cluster_lang_feat, label_feat) + 1
    labeled_assignments = cluster_to_label[assignments]
   
    aligned_xyz = align_to_gt(xyz.cpu(), points, scene_name)
    print("Groundtruth points extents:")
    print("min:", points.min(axis=0), "max:", points.max(axis=0))
    print("Means3D extents:")
    print("min:", aligned_xyz.min(axis=0), "max:", aligned_xyz.max(axis=0))
    kd_tree = cKDTree(aligned_xyz)
    dist, nn_idx = kd_tree.query(points, k=1)
    print("Nearest neighbor distance statistics:")
    print("Min distance:", dist.min())
    print("Max distance:", dist.max())
    print("Mean distance:", dist.mean())

    pred_labels = labeled_assignments[nn_idx]  # shape [num_vertices]
    # pred_labels_t = torch.from_numpy(pred_labels).long().cuda()

    # Calculate metrics
    _, mean_iou, accuracy, mean_class_accuracy = calculate_metrics(gt_labels, pred_labels)
    print(f"[{scene_name}] mIoU={mean_iou:.4f}, "
            f"Overall Acc={accuracy:.4f}, "
            f"MeanClassAcc={mean_class_accuracy:.4f}")

    # ###
    # unlabeled_mask = (new_labels == 0)
    # correct_mask   = (new_labels == pred_labels) & ~unlabeled_mask

    # # 2) Create an array of colors, one for each vertex
    # colors = np.zeros((points.shape[0], 3), dtype=np.float32)  # default is black (0,0,0)

    # # 3) Color correct predictions (green)
    # colors[correct_mask] = [0.0, 1.0, 0.0]   # green

    # # 4) Color all other labeled points that are incorrect (red)
    # incorrect_mask = (~correct_mask) & (~unlabeled_mask)
    # colors[incorrect_mask] = [1.0, 0.0, 0.0]  # red
    
    # # 3) Save the point cloud to a PLY file
    # output_ply = f"benchmarking/{scene_name}_prediction_visual.ply"
    # save_colored_ply(output_ply, points, colors)

def align_to_gt(xyz, gt_xyz, scene_name):
    # manual initial guess (r_x, r_y, r_z, t_x, t_y, t_z)
    transformations = {
        "scene0050_00": (-140, 10, 215, 5, 2, 2)
    }
    (r_x, r_y, r_z, t_x, t_y, t_z) = transformations[scene_name]
    
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt_xyz)
    pcd_gt.paint_uniform_color([0, 1, 0])  # Green for GT

    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(xyz)    
    pred_pcd.paint_uniform_color([1, 0, 0])  # Red for Gaussians

    rotation_matrix = R.from_euler('xyz', [r_x, r_y, r_z], degrees=True).as_matrix()
    rotation_4x4 = np.eye(4)
    rotation_4x4[:3, :3] = rotation_matrix
    translation_4x4 = np.eye(4)
    translation_4x4[:3, 3] = [t_x, t_y, t_z]

    pred_pcd.transform(rotation_4x4)
    pred_pcd.transform(translation_4x4)
    
    # --- Automatic Alignment Using ICP ---
    print("Starting automatic alignment...")
    dTau = 0.005
    trajectory_transform = np.eye(4)

    # Perform coarse alignment
    r2 = registration_vol_ds(pred_pcd, pcd_gt, trajectory_transform, dTau, dTau * 50, 100)
    print("Stage 2 transformation:\n", r2.transformation)

    # Apply the transformation
    pcd_aligned = pred_pcd.transform(r2.transformation)

    # --- Extract Transformed XYZ to PyTorch Tensor ---
    transformed_xyz = np.asarray(pcd_aligned.points)  # Convert to NumPy array
    transformed_tensor = torch.tensor(transformed_xyz, dtype=torch.float32)  # Convert to PyTorch tensor
    print("Extracted transformed XYZ coordinates into a tensor:", transformed_tensor.shape)

    return transformed_tensor


def registration_vol_ds(source, target, init_transformation, dTau, threshold, iterations):
    s_down = source.voxel_down_sample(dTau)
    t_down = target.voxel_down_sample(dTau)
    t_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30)
    )
    reg = o3d.pipelines.registration.registration_icp(
        s_down, t_down, threshold, init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations)
    )
    reg.transformation = reg.transformation @ init_transformation
    return reg