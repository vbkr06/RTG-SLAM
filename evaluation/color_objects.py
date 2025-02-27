import torch
import numpy as np
import torch
import torch.nn.functional as F
import clip
from evaluation.replica_constants import OVOSLAM_COLORED_LABELS, LABEL_TO_COLOR
import open3d as o3d
import os

# input : gaussians xyz, assignments [gaussians, clusters], cluster language features, label set
# flatten assignments to 1D by selecting best cluster (closest clip distance to mean)
# assign label to gaussians
# color gaussians according to label


def label_cluster_features(cluster_lang_features, label_features, device="cuda:0"):
    label_features = label_features.to(device)
    cluster_lang_features_tensor = cluster_lang_features.half().to(device)
    label_features = F.normalize(label_features, p=2, dim=1)
    cluster_features_normalized = F.normalize(cluster_lang_features_tensor, p=2, dim=1)

    similarities = cluster_features_normalized @ label_features.t()

    cluster_best_label_ids = torch.argmax(similarities, dim=1)
    return cluster_best_label_ids

def load_text_embeddings(dataset="Replica"):
    if dataset == "Replica":
        class_names = OVOSLAM_COLORED_LABELS
    elif dataset == "Scannet":
        class_names = []
    
    clip_model, _ = clip.load("ViT-B/32", device="cuda")
    text_prompts = class_names
    text_tokens = clip.tokenize(text_prompts).cuda()
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  
    return class_names, text_features
import torch
import torch.nn.functional as F

def pick_best_cluster_with_masks(
    assignments: torch.Tensor,        # [N, C] bool
    cluster_masks: dict,             # {cluster_id: [(frame_id, mask_id), ...]}
    mask_lang_feat: dict,            # mask_lang_feat[fid][mid] => Tensor[emb_dim]
    emb_dim: int = 512,
    device: str = "cuda:0"
) -> torch.Tensor:
    """
    For each Gaussian (row in assignments), gather all masks from the candidate clusters,
    compute pairwise similarities among those mask features, pick the single mask with
    the highest total similarity, and return the cluster from which that mask originated.

    Args:
      assignments: [N, C] boolean, True if gaussian i is assigned to cluster c.
      cluster_masks: dict c -> list of (frame_id, mask_id).
      mask_lang_feat: mask_lang_feat[fid][mid] => 512-dim feature.
      device: which device to use.

    Returns:
      best_cluster: [N] long tensor of cluster IDs (or -1 if none).
    """
    N, C = assignments.shape
    best_cluster = torch.full((N,), -1, dtype=torch.long, device=device)
    best_cluster_feat = torch.zeros((N, emb_dim), dtype=torch.float, device=device)
    
    for i in range(N):
        candidate_clusters = torch.nonzero(assignments[i], as_tuple=True)[0]
        if candidate_clusters.numel() == 0:
            continue

        union_feats_list = []
        union_cluster_ids = []

        for c in candidate_clusters:
            c_int = c.item()
            if c_int not in cluster_masks:
                continue
            for (fid, mid) in cluster_masks[c_int]:
                feat = mask_lang_feat[fid][mid].to(device)
                union_feats_list.append(feat)
                union_cluster_ids.append(c_int)

        if not union_feats_list:
            continue

        union_feats = torch.stack(union_feats_list, dim=0)  # [U, emb_dim]
        union_norm = F.normalize(union_feats, p=2, dim=1)   # [U, emb_dim]
        sim = union_norm @ union_norm.t()                   # [U, U]
        row_scores = sim.sum(dim=1)                         # [U]
        best_idx = torch.argmax(row_scores)
        best_cluster[i] = union_cluster_ids[best_idx]
        best_cluster_feat[i] = union_feats[best_idx]

    return best_cluster, best_cluster_feat


def map_gaussian_labels(best_cluster, cluster_lang_feat, device="cuda:0"):
    _, text_features = load_text_embeddings()
    cluster_label_ids = label_cluster_features(cluster_lang_feat, text_features, device=device)
    gaussian_labels = torch.where(
        best_cluster == -1,
        torch.full_like(best_cluster, -1),
        cluster_label_ids[best_cluster]
    )
    return gaussian_labels

def process_gaussians(assignments, cluster_masks, mask_lang_feat, device="cuda:0"):
    best_cluster, cluster_lang_feat = pick_best_cluster_with_masks(assignments, cluster_masks, mask_lang_feat)
    gaussian_labels = map_gaussian_labels(best_cluster, cluster_lang_feat, device=device)
    return best_cluster, gaussian_labels


def save_colored_objects_ply(means_xyz, assignments, cluster_masks, mask_lang_feat, ply_filename="visualization/colored_scene/colored_gaussians.ply"):
    _, gaussian_labels = process_gaussians(means_xyz, assignments, cluster_masks, mask_lang_feat)

    means_xyz_np = means_xyz.detach().cpu().numpy()

    labels_cpu = gaussian_labels.detach().cpu()  # shape: [N]
    mask_no_label = (labels_cpu == -1)
    
    clamped_labels = torch.clamp(labels_cpu, min=0)
    
    colors_tensor = LABEL_TO_COLOR[clamped_labels]  # shape: [N, 3]
    colors_tensor /= 255.0    
    colors_tensor[mask_no_label] = 0
    colors_np = colors_tensor.numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means_xyz_np)
    pcd.colors = o3d.utility.Vector3dVector(colors_np)
    
    os.makedirs(os.path.dirname(ply_filename), exist_ok=True)
    o3d.io.write_point_cloud(ply_filename, pcd)