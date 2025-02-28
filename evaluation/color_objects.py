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

# def pick_best_cluster_with_masks(
#     assignments: torch.Tensor,        # [N, C] bool
#     cluster_masks: dict,             # {cluster_id: [(frame_id, mask_id), ...]}
#     mask_lang_feat: dict,            # mask_lang_feat[fid][mid] => Tensor[emb_dim]
#     emb_dim: int = 512,
#     device: str = "cuda:0"
# ) -> torch.Tensor:
#     """
#     For each Gaussian (row in assignments), gather all masks from the candidate clusters,
#     compute pairwise similarities among those mask features, pick the single mask with
#     the highest total similarity, and return the cluster from which that mask originated.

#     Args:
#       assignments: [N, C] boolean, True if gaussian i is assigned to cluster c.
#       cluster_masks: dict c -> list of (frame_id, mask_id).
#       mask_lang_feat: mask_lang_feat[fid][mid] => 512-dim feature.
#       device: which device to use.

#     Returns:
#       best_cluster: [N] long tensor of cluster IDs (or -1 if none).
#     """
#     N, C = assignments.shape
#     best_cluster = torch.full((N,), -1, dtype=torch.long, device=device)
#     best_cluster_feat = torch.zeros((N, emb_dim), dtype=torch.float, device=device)
    
#     for i in range(N):
#         candidate_clusters = torch.nonzero(assignments[i], as_tuple=True)[0]
#         if candidate_clusters.numel() == 0:
#             continue

#         union_feats_list = []
#         union_cluster_ids = []

#         for c in candidate_clusters:
#             c_int = c.item()
#             for (fid, mid) in cluster_masks[c_int]:
#                 feat = mask_lang_feat[fid][mid].to(device)
#                 union_feats_list.append(feat)
#                 union_cluster_ids.append(c_int)

#         if not union_feats_list:
#             continue

#         union_feats = torch.stack(union_feats_list, dim=0)  # [U, emb_dim]
#         union_norm = F.normalize(union_feats, p=2, dim=1)   # [U, emb_dim]
#         sim = union_norm @ union_norm.t()                   # [U, U]
#         row_scores = sim.sum(dim=1)                         # [U]
#         best_idx = torch.argmax(row_scores)
#         best_cluster[i] = union_cluster_ids[best_idx]
#         best_cluster_feat[i] = union_feats[best_idx]

#     return best_cluster, best_cluster_feat
def pick_best_cluster_with_masks(
    assignments: torch.Tensor,
    cluster_masks: dict,
    mask_lang_feat: dict,
    emb_dim: int = 512,
    device: str = "cuda:0",
    max_masks: int = 20,
    max_candidate_clusters: int = 5,
    batch_size: int = 10000
):
    N, C = assignments.shape

    cluster_feats = torch.zeros((C, max_masks, emb_dim), device=device)
    cluster_valid = torch.zeros((C, max_masks), dtype=torch.bool, device=device)
    for c in range(C):
        masks = cluster_masks.get(c, [])
        for j, (fid, mid) in enumerate(masks[:max_masks]):
            cluster_feats[c, j] = mask_lang_feat[fid][mid].to(device)
            cluster_valid[c, j] = True

    best_clusters_list = []
    best_cluster_feats_list = []

    for i in range(0, N, batch_size):
        batch_assignments = assignments[i:i+batch_size]
        B = batch_assignments.shape[0]

        candidate_tensor = torch.zeros((B, max_candidate_clusters, max_masks, emb_dim), device=device)
        candidate_valid = torch.zeros((B, max_candidate_clusters, max_masks), dtype=torch.bool, device=device)
        candidate_cluster_idx = -torch.ones((B, max_candidate_clusters), dtype=torch.long, device=device)

        for b in range(B):
            cand_indices = torch.nonzero(batch_assignments[b], as_tuple=False).view(-1)
            num_candidates = min(cand_indices.numel(), max_candidate_clusters)
            if num_candidates > 0:
                candidate_tensor[b, :num_candidates] = cluster_feats[cand_indices[:num_candidates]]
                candidate_valid[b, :num_candidates] = cluster_valid[cand_indices[:num_candidates]]
                candidate_cluster_idx[b, :num_candidates] = cand_indices[:num_candidates]

        cand = candidate_tensor.view(B, max_candidate_clusters * max_masks, emb_dim)
        valid = candidate_valid.view(B, max_candidate_clusters * max_masks)

        cand_norm = F.normalize(cand, p=2, dim=-1) * valid.unsqueeze(-1).float()
        S = cand_norm.sum(dim=1)
        scores = (cand_norm * S.unsqueeze(1)).sum(dim=-1)
        scores[~valid] = -float('inf')

        best_idx = scores.argmax(dim=1)
        best_cluster_feat_batch = cand[torch.arange(B), best_idx]
        best_candidate_slot = best_idx // max_masks
        best_cluster_batch = candidate_cluster_idx[torch.arange(B), best_candidate_slot]

        no_valid = valid.sum(dim=1) == 0
        best_cluster_batch[no_valid] = -1
        best_cluster_feat_batch[no_valid] = 0

        best_clusters_list.append(best_cluster_batch)
        best_cluster_feats_list.append(best_cluster_feat_batch)

    best_cluster = torch.cat(best_clusters_list, dim=0)
    best_cluster_feat = torch.cat(best_cluster_feats_list, dim=0)
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
    _, gaussian_labels = process_gaussians(assignments, cluster_masks, mask_lang_feat)

    means_xyz = means_xyz.cpu().numpy() if torch.is_tensor(means_xyz) else means_xyz
    colors = np.zeros((len(gaussian_labels), 3), dtype=np.float32)

    for i in range(len(gaussian_labels)):
        label_id = gaussian_labels[i].item() if isinstance(gaussian_labels[i], torch.Tensor) else gaussian_labels[i]
        if label_id == -1:
            colors[i] = (0, 0, 0)
        else:
            label_name = OVOSLAM_COLORED_LABELS[label_id]
            rgb = LABEL_TO_COLOR.get(label_name, (0, 0, 0))
            colors[i] = np.array(rgb) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means_xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    os.makedirs(os.path.dirname(ply_filename), exist_ok=True)
    o3d.io.write_point_cloud(ply_filename, pcd)
