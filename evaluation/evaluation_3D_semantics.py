import torch
import os
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement
import torch.nn.functional as F
import clip
import open3d as o3d
from evaluation.constants import OVOSLAM_COLORED_LABELS, NYU40, NYU20_INDICES, REPLICA_EXISTING_CLASSES, REPLICA_CLASSES
import json


def read_labels_from_scannet_ply(file_path):
    ply_data = PlyData.read(file_path)
    vertex_data = ply_data['vertex'].data
    points = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
    labels = vertex_data['label']  
    return points, labels

def load_instance_to_class_map(json_path):
    """
    Reads Habitat's `info_semantic.json` and extracts a mapping from object ID → class ID.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Create a mapping {object_id: class_id}
    instance_to_class = {obj["id"]: obj["class_id"] for obj in data["objects"]}
    return instance_to_class

def create_object_to_subset_class_map(json_path, replica_classes, instance_to_class):
    class_to_subset = {class_id: i for i, class_id in enumerate(replica_classes)}

    # Build a mapping: object id → subset class id.
    object_to_subset_class = {}
    for obj_id, class_id in instance_to_class.items():
        subset_class = class_to_subset.get(class_id, -1)
        object_to_subset_class[obj_id] = subset_class

    return object_to_subset_class

def read_from_replica_ply(file_path, json_path, num_classes=51):
    instance_to_class = load_instance_to_class_map(json_path)
    class_indices = []
    if num_classes == 101:
        class_indices = [i for i in range(len(REPLICA_CLASSES))]
    elif num_classes == 51:
        class_indices = REPLICA_EXISTING_CLASSES

    object_to_subset_class = create_object_to_subset_class_map(json_path, class_indices, instance_to_class)

    ply_data = PlyData.read(file_path)
    vertex_data = ply_data['vertex'].data

    points = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T

    num_points = len(points)
    class_ids = np.full(num_points, -1, dtype=np.int32)
    subset_class_ids = np.full(num_points, -1, dtype=np.int32)

    if 'face' in ply_data and 'object_id' in ply_data['face'].data.dtype.names:
        face_data = ply_data['face'].data
        face_object_ids = np.array(face_data['object_id'])  # (num_faces,)
        face_vertex_indices = face_data['vertex_indices']     # List of lists

        for i, indices in enumerate(face_vertex_indices):
            object_id = face_object_ids[i]  # Get object ID for this face
            full_class = instance_to_class.get(object_id, -1)
            subset_class = object_to_subset_class.get(object_id, -1)

            idx = np.array(indices)
            class_ids[idx] = full_class
            subset_class_ids[idx] = subset_class

    return points, class_ids, subset_class_ids



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

# def label_cluster_features(cluster_lang_features, label_features, device="cuda:0"):
#     label_features = label_features.to(device)
#     cluster_lang_features_tensor = cluster_lang_features.half().to(device)
#     label_features = F.normalize(label_features, p=2, dim=1)
#     cluster_features_normalized = F.normalize(cluster_lang_features_tensor, p=2, dim=1)

#     similarities = cluster_features_normalized @ label_features.t()

#     cluster_best_label_ids = torch.argmax(similarities, dim=1)
#     return cluster_best_label_ids

def label_cluster_features_batched(cluster_lang_features, label_features, batch_size=10, device="cuda:0"):
    label_features = label_features.to(device)
    cluster_lang_features = cluster_lang_features.half().to(device)
    
    # Normalize label features once
    label_features = F.normalize(label_features, p=2, dim=1)

    num_clusters = cluster_lang_features.shape[0]
    cluster_best_label_ids = torch.empty(num_clusters, dtype=torch.long, device=device)

    for start in range(0, num_clusters, batch_size):
        end = min(start + batch_size, num_clusters)

        # Extract batch and normalize
        cluster_batch = cluster_lang_features[start:end]
        cluster_batch = F.normalize(cluster_batch, p=2, dim=1)

        # Compute similarities
        similarities = cluster_batch @ label_features.T

        # Get best labels
        cluster_best_label_ids[start:end] = torch.argmax(similarities, dim=1)

    return cluster_best_label_ids


# def load_text_embeddings(labels):
#     clip_model, _ = clip.load("ViT-B/32", device="cuda")
#     text_prompts = labels
#     text_tokens = clip.tokenize(text_prompts).cuda()
#     with torch.no_grad():
#         text_features = clip_model.encode_text(text_tokens)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)  
#     return text_features

def load_text_embeddings(labels, batch_size=32):
    clip_model, _ = clip.load("ViT-B/32", device="cuda")
    clip_model.eval()
    
    all_features = []
    with torch.no_grad():
        # Process labels in batches.
        for i in range(0, len(labels), batch_size):
            batch = labels[i:i+batch_size]
            text_tokens = clip.tokenize(batch).to("cuda")
            text_features = clip_model.encode_text(text_tokens)
            # Normalize the text features.
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            all_features.append(text_features)
    
    # Concatenate all batched features.
    text_features = torch.cat(all_features, dim=0)
    return text_features

def save_colored_ply(filepath, xyz, rgb):
    if rgb.dtype != np.uint8:
        rgb_255 = (rgb * 255).clip(0, 255).astype(np.uint8)
    else:
        rgb_255 = rgb

    num_verts = xyz.shape[0]

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
    output_dir = '/mnt/scratch/cluster_simple_ply'

    # target_id_mapping = {}
    # for new_idx, old_label in enumerate(NYU40.keys(), start=1):
    #     target_id_mapping[old_label] = new_idx
    # new_labels = np.zeros_like(labels, dtype=np.int32)
    # for old_value, new_value in target_id_mapping.items():
    #     mask = (labels == old_value)
    #     new_labels[mask] = new_value
    gt_points, gt_labels = None, None
    if dataset == "Replica":
        gt_ply = os.path.join(dataset_params.source_path, "mesh_semantic.ply")
        json_path = os.path.join(dataset_params.source_path, "info_semantic.json")
        gt_points, _, subset_ids = read_from_replica_ply(gt_ply, json_path)
        gt_labels = torch.from_numpy(subset_ids).long().cuda()
        text_labels = [REPLICA_CLASSES[i] for i in REPLICA_EXISTING_CLASSES][1:]
    elif dataset == "Scannetpp":
        gt_ply = os.path.join(dataset_params.source_path, f"{scene_name}_vh_clean_2.labels.ply")
        gt_points, labels = read_labels_from_scannet_ply(gt_ply)
        gt_labels = torch.from_numpy(np.array(labels, dtype=np.int32)).long().cuda()
        text_labels = list(NYU40.values())[1:]#[NYU40[i] for i in NYU20_INDICES]

    print(f"[{scene_name}] Loaded {gt_points.shape[0]} vertices. Max label={gt_labels.max()}")

    label_feat = load_text_embeddings(text_labels)
    cluster_to_label = label_cluster_features_batched(cluster_lang_feat, label_feat) + 1
    labeled_assignments = cluster_to_label[assignments]
   
    aligned_xyz = align_to_gt(xyz.cpu(), gt_points, scene_name)
    print("Groundtruth points extents:")
    print("min:", gt_points.min(axis=0), "max:", gt_points.max(axis=0))
    print("Means3D extents:")
    print("min:", aligned_xyz.min(axis=0), "max:", aligned_xyz.max(axis=0))
    kd_tree = cKDTree(aligned_xyz)
    dist, nn_idx = kd_tree.query(gt_points, k=1)
    print("Nearest neighbor distance statistics:")
    print("Min distance:", dist.min())
    print("Max distance:", dist.max())
    print("Mean distance:", dist.mean())

    pred_labels = labeled_assignments[nn_idx]  # shape [num_vertices]
    # pred_labels_t = torch.from_numpy(pred_labels).long().cuda()

    # Calculate metrics
    ious, mean_iou, accuracy, mean_class_accuracy = calculate_metrics(gt_labels, pred_labels)
    result_str = f"[{scene_name}] mIoU={mean_iou:.4f}, Overall Acc={accuracy:.4f}, MeanClassAcc={mean_class_accuracy:.4f}\n"
    print(result_str)
    with open("metrics.txt", "a") as f:
        f.write(result_str)
        present_classes = torch.unique(gt_labels.cpu())
        present_classes = [int(cls) for cls in present_classes if int(cls) != 0]
        class_iou_list = []
        for cls in present_classes:
            iou_val = ious[cls].item()
            class_name = text_labels[cls - 1] if cls - 1 < len(text_labels) else f"Class {cls}"
            class_iou_list.append((class_name, iou_val))
        class_iou_list.sort(key=lambda x: x[1], reverse=True)
        for class_name, iou_val in class_iou_list:
            f.write(f"{class_name} IoU: {iou_val:.4f}\n")

    ###
    pred_labels_np = pred_labels.cpu().numpy()
    gt_labels_np = gt_labels.cpu().numpy()

    # Create boolean masks using the NumPy array.
    unlabeled_mask = (pred_labels_np == 0)
    correct_mask   = (pred_labels_np == gt_labels_np) & ~unlabeled_mask

    colors = np.zeros((gt_points.shape[0], 3), dtype=np.float32)  # default is black (0,0,0)

    # Correct predictions (green)
    colors[correct_mask] = [0.0, 1.0, 0.0]

    # Incorrect predictions (red)
    incorrect_mask = (~correct_mask) & (~unlabeled_mask)
    colors[incorrect_mask] = [1.0, 0.0, 0.0]

    output_ply = f"{output_dir}/{scene_name}_prediction.ply"
    save_colored_ply(output_ply, gt_points, colors)

def align_to_gt(xyz, gt_xyz, scene_name):
    # manual initial guess (r_x, r_y, r_z, t_x, t_y, t_z)
    transformations = {
        "scene0011_00": (-140, 10, 215, 5, 2, 2),           # dummy
        "scene0050_00": (-140, 10, 215, 5, 2, 2),           # aligned
        "scene0231_00": (-140, 10, 215, 5, 2, 2),           # dummy
        "scene0378_00": (-140, 10, 215, 5, 2, 2),           # dummy
        "scene0518_00": (-140, 10, 215, 5, 2, 2),           # dummy
        "room0":        (-110, -5, 110, 3.5, 0.2, 0.3),     # aligned
        "room1":        (-110, -5, 110, 3.5, 0.2, 0.3),     # dummy
        "room2":        (-110, -5, 110, 3.5, 0.2, 0.3),     # dummy
        "office0":        (-110, -5, 110, 3.5, 0.2, 0.3),   # dummy
        "office1":        (-110, -5, 110, 3.5, 0.2, 0.3),   # dummy
        "office2":        (-110, -5, 110, 3.5, 0.2, 0.3),   # dummy
        "office3":        (-110, -5, 110, 3.5, 0.2, 0.3),   # dummy
        "office4":        (-110, -5, 110, 3.5, 0.2, 0.3),   # dummy
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
    
    print("Starting automatic alignment...")
    dTau = 0.005
    trajectory_transform = np.eye(4)

    r2 = registration_vol_ds(pred_pcd, pcd_gt, trajectory_transform, dTau, dTau * 50, 100)
    print("Automatic Transformation:\n", r2.transformation)

    pcd_aligned = pred_pcd.transform(r2.transformation)
    o3d.io.write_point_cloud("/mnt/projects/FeatureGSLAM/ply/room0.ply", pcd_aligned)

    transformed_xyz = np.asarray(pcd_aligned.points) 
    transformed_tensor = torch.tensor(transformed_xyz, dtype=torch.float32)  
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


def save_evaluation_data(stable_xyz, stable_assignments, oneD_assignments, cluster_lang_feat, dataset_params, save_path):
    data = {
        "xyz": stable_xyz.cpu() if hasattr(stable_xyz, "cpu") else stable_xyz,
        "stable_assignments": stable_assignments.cpu() if hasattr(stable_assignments, "cpu") else stable_assignments,
        "oneD_assignments": oneD_assignments.cpu() if hasattr(oneD_assignments, "cpu") else oneD_assignments,
        "cluster_lang_feat": cluster_lang_feat.cpu() if hasattr(cluster_lang_feat, "cpu") else cluster_lang_feat,
        "dataset": dataset_params
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(data, save_path)
    print(f"Saved evaluation data to {save_path}")


import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D Semantics from saved evaluation data.")
    parser.add_argument("--data_file", type=str, default="/mnt/projects/FeatureGSLAM/Replica/room0/evaluation/3Dsemantic.pt",
                        help="Path to the saved evaluation data file (e.g. /mnt/scratch/cluster_evaluation_data.pt).")
    args = parser.parse_args()

    data = torch.load(args.data_file)
    print(f"Loaded evaluation data from {args.data_file}")

    xyz = data["xyz"]
    assignments = data["oneD_assignments"]  
    cluster_lang_feat = data["cluster_lang_feat"]
    dataset_params = data["dataset"]  

    evaluate_3D_semantics(xyz, assignments, cluster_lang_feat, dataset_params)

if __name__ == "__main__":
    main()