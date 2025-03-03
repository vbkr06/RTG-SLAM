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
    total_classes = gt.max().item()
    gt = gt.cpu()
    pred = pred.cpu()

    pred[gt == 0] = 0  # ignore unlabeled in prediction

    intersection = torch.zeros(total_classes)
    union        = torch.zeros(total_classes)
    correct      = torch.zeros(total_classes)
    total        = torch.zeros(total_classes)

    for cls_id in range(1, total_classes + 1):
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
    """
    Save points with colors to a PLY file using plyfile.
    :param filepath: Output path, e.g. 'result.ply'
    :param xyz: (N,3) NumPy array of point coordinates
    :param rgb: (N,3) NumPy array of RGB in [0..1], or [0..255]
    """
    # Ensure rgb is in 0..255 range (uint8) for standard PLY viewers
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

    # # Apply your scene-specific transformation
    # means3D = rotate_points(means3D, 250, 'x')
    # means3D = rotate_points(means3D, 0, 'y')
    # means3D = rotate_points(means3D, 159, 'z')
    # means3D = translate_points(means3D, [2.76, 2.98, 1.3])

    # cluster_to_label = data['assigned_label_idx']
    # gaussian_labels = cluster_to_label[gaussian_cluster_assignments]

    # print("Groundtruth points extents:")
    # print("min:", points.min(axis=0), "max:", points.max(axis=0))
    # print("Means3D extents:")
    # print("min:", means3D.min(axis=0), "max:", means3D.max(axis=0))
    
    label_feat = load_text_embeddings(text_labels)
    cluster_to_label = label_cluster_features(cluster_lang_feat, label_feat) + 1
    labeled_assignments = cluster_to_label[assignments]
    # # Remap predicted labels
    # if len(gaussian_labels.shape) > 1:
    #     gaussian_labels = gaussian_labels.squeeze(-1)

    # # Use nearest-neighbor to assign each vertex in GT to a predicted label
    kd_tree = cKDTree(xyz.cpu())
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