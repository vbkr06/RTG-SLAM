import torch


def calculate_metrics(gt, pred, total_classes=20):
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

    # Classes actually appearing in GT
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