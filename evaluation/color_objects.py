import torch
import numpy as np
import torch
import torch.nn.functional as F
import clip
from evaluation.constants import OVOSLAM_COLORED_LABELS, LABEL_TO_COLOR, NYU40, NYU20_INDICES
import open3d as o3d
import os
import random
import matplotlib.pyplot as plt

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

def load_text_embeddings(dataset):
    if dataset == "Replica":
        class_names = OVOSLAM_COLORED_LABELS
    elif dataset == "Scannetpp":
        class_names = list(NYU40.values())[1:]
    
    clip_model, _ = clip.load("ViT-B/32", device="cuda")
    text_prompts = class_names
    text_tokens = clip.tokenize(text_prompts).cuda()
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  
    return class_names, text_features


def get_cluster_language_features(cluster_masks, mask_language_features, num_clusters, max_samples=100, emb_dim=512, device="cuda"):
    cluster_lang_features = torch.zeros((num_clusters, emb_dim), device=device)

    for cluster_idx, all_mask_infos in cluster_masks.items():
        num_masks = len(all_mask_infos)
        if num_masks == 0:
            continue
        
        sampled_masks = random.sample(all_mask_infos, min(max_samples, num_masks))
        all_features_curr_cluster = torch.zeros((len(sampled_masks), emb_dim), device=device)

        for i, (mask_time_idx, mask_index) in enumerate(sampled_masks):
            all_features_curr_cluster[i] = mask_language_features[mask_time_idx][mask_index]

        similarity_matrix = F.cosine_similarity(
            all_features_curr_cluster.unsqueeze(1), 
            all_features_curr_cluster.unsqueeze(0), 
            dim=-1
        )
        similarity_matrix.fill_diagonal_(0)
        mean_similarities = similarity_matrix.mean(dim=1)
        best_index = mean_similarities.argmax()

        cluster_lang_features[cluster_idx] = all_features_curr_cluster[best_index]

    return cluster_lang_features


# def save_colored_objects_ply_simple(frame_id, means_xyz, assignments, cluster_labels, ply_dir="/mnt/scratch/cluster_simple_ply"):
#     os.makedirs(ply_dir, exist_ok=True)

#     if torch.is_tensor(means_xyz):
#         means_xyz = means_xyz.cpu().numpy()
#     if torch.is_tensor(assignments):
#         assignments = assignments.cpu().numpy()

#     C = assignments.shape[1]
#     np.random.seed(42)
#     cluster_colors = np.random.randint(0, 255, (C, 3), dtype=np.uint8) / 255.0
#     all_colors = np.zeros((means_xyz.shape[0], 3))

#     for c in range(C):
#         cluster_indices = np.where(assignments[:, c])[0]
#         if len(cluster_indices) == 0:
#             continue
        
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(means_xyz[cluster_indices])
#         pcd.colors = o3d.utility.Vector3dVector(np.tile(cluster_colors[c], (len(cluster_indices), 1)))

#         all_colors[cluster_indices] = cluster_colors[c]


#         # ply_filename = os.path.join(ply_dir, f"{frame_id}_{c}.ply")
#         # o3d.io.write_point_cloud(ply_filename, pcd)

#     total_pcd = o3d.geometry.PointCloud()
#     total_pcd.points = o3d.utility.Vector3dVector(means_xyz)
#     total_pcd.colors = o3d.utility.Vector3dVector(all_colors)
#     ply_filename_complete = os.path.join(ply_dir, f"complete_scene_{frame_id}.ply")
#     o3d.io.write_point_cloud(ply_filename_complete, total_pcd)
    
#     print(f"Saved cluster .ply files to {ply_dir}")
def save_colored_objects_ply_simple(frame_id, means_xyz, assignments, cluster_masks, mask_lang_feat, dataset="Replica", ply_dir="/mnt/scratch/cluster_simple_ply"):
    os.makedirs(ply_dir, exist_ok=True)
    if torch.is_tensor(means_xyz):
        means_xyz = means_xyz.cpu().numpy()
    if torch.is_tensor(assignments):
        assignments = assignments.cpu().numpy()
    N, C = assignments.shape
    
    # Get language features and label IDs first
    cluster_lang_features = get_cluster_language_features(cluster_masks, mask_lang_feat, C, emb_dim=512, device="cuda")
    class_names, text_features = load_text_embeddings(dataset=dataset)
    cluster_best_label_ids = label_cluster_features(cluster_lang_features, text_features, device="cuda")
    
    # # Create a mapping from label ID to color, ensuring same labels get same colors
    # np.random.seed(42)
    # unique_label_ids = list(set([label_id.item() for label_id in cluster_best_label_ids]))
    label_to_color = {
        label_id: np.array(LABEL_TO_COLOR[OVOSLAM_COLORED_LABELS[label_id]]) / 255.0
        for label_id in cluster_best_label_ids
    }
    # Assign colors based on label IDs, not cluster IDs
    cluster_colors = np.zeros((C, 3))
    for c in range(C):
        label_id = cluster_best_label_ids[c].item()
        cluster_colors[c] = label_to_color[label_id]
    
    # Apply colors to points
    all_colors = np.zeros((N, 3))
    for c in range(C):
        inds = np.where(assignments[:, c])[0]
        if len(inds) > 0:
            all_colors[inds] = cluster_colors[c]
    
    # Create point cloud
    total_pcd = o3d.geometry.PointCloud()
    total_pcd.points = o3d.utility.Vector3dVector(means_xyz)
    total_pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    # Set point size to be subtle (Open3D doesn't directly control point size in the PLY,
    # but we can use voxel downsampling to create cleaner points)
    # Note: The actual point size will be controlled when visualizing the PLY file
    
    ply_filename = os.path.join(ply_dir, f"complete_scene_{frame_id}.ply")
    o3d.io.write_point_cloud(ply_filename, total_pcd)
    
    # Create legend based on label IDs, not cluster IDs
    legend_items = {}  # Use a dict to avoid duplicates for the same label
    for c in range(C):
        if np.sum(assignments[:, c]) > 0:
            label_idx = cluster_best_label_ids[c].item()
            class_name = class_names[label_idx]
            # Only add this label once to the legend
            if class_name not in legend_items:
                legend_items[class_name] = (label_idx, cluster_colors[c])
    
    # Convert to lists for plotting
    legend_labels = [f"{class_name} (Label ID: {label_idx})" for class_name, (label_idx, _) in legend_items.items()]
    legend_colors = [color for _, (_, color) in legend_items.items()]
    
    fig, ax = plt.subplots(figsize=(6, len(legend_labels)*0.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(legend_labels))
    for i, (lbl, col) in enumerate(zip(legend_labels, legend_colors)):
        ax.add_patch(plt.Rectangle((0, i), 1, 1, color=col, ec='black'))
        ax.text(1.05, i+0.5, lbl, va='center', fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    legend_filename = os.path.join(ply_dir, f"legend.png")
    plt.savefig(legend_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved complete scene to {ply_filename}")
    print(f"Saved legend to {legend_filename}")