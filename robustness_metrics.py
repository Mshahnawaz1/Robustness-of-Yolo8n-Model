import numpy as np
from typing import Tuple

# --- Helper Function 1: Intersection over Union (IoU) ---

def calculate_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Boxes are expected in (x_min, y_min, x_max, y_max) format.
    """
    # Determine the coordinates of the intersection rectangle
    x_min_inter = np.maximum(boxA[0], boxB[0])
    y_min_inter = np.maximum(boxA[1], boxB[1])
    x_max_inter = np.minimum(boxA[2], boxB[2])
    y_max_inter = np.minimum(boxA[3], boxB[3])

    # Calculate area of intersection rectangle
    intersection_width = np.maximum(0, x_max_inter - x_min_inter)
    intersection_height = np.maximum(0, y_max_inter - y_min_inter)
    area_intersection = intersection_width * intersection_height

    # Calculate area of both the prediction and ground-truth rectangles
    area_A = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_B = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Calculate the area of union
    area_union = area_A + area_B - area_intersection

    # Compute the IoU
    iou = area_intersection / area_union if area_union > 0 else 0.0
    return iou


# --- Main Function: Calculate Simple Robustness Metrics ---

def calculate_robustness_metrics(
    gt_results,
    pred_results, 
    iou_threshold: float = 0.5
) -> Tuple[float, float, int]:
    """
    Calculates two simple metrics for robustness:
    1. Overall Recall (Detection Rate)
    2. Average IoU of all True Positive detections
    
    Args:
        gt_results (YOLO results object): The detection results from the clean image (Pseudo-GT).
        pred_results (YOLO results object): The detection results from the degraded image (Prediction).
        iou_threshold (float): The IoU threshold for counting a True Positive.
        
    Returns:
        Tuple[float, float, int]: (Overall Recall, Average IoU of TPs, True Positives Count)
    """
        
    if not gt_results or not gt_results[0].boxes:
        # Cannot calculate if no Ground Truth exists
        return 0.0, 0.0, 0
    
    gt_boxes = gt_results[0].boxes.xyxy.cpu().numpy()
    
    # Predictions must be sorted by score for matching logic
    if not pred_results or not pred_results[0].boxes:
        return 0.0, 0.0, 0

    pred_boxes_raw = pred_results[0].boxes.xyxy.cpu().numpy()
    pred_scores_raw = pred_results[0].boxes.conf.cpu().numpy()

    num_gt = len(gt_boxes)
    num_pred = len(pred_boxes_raw)
    
    if num_gt == 0:
        return 0.0, 0.0, 0
    
    # Sort predictions by confidence score in descending order
    sorted_indices = np.argsort(pred_scores_raw)[::-1]
    pred_boxes = pred_boxes_raw[sorted_indices]
    
    gt_matched = np.zeros(num_gt, dtype=bool)
    true_positive_ious = []
    
    # --- 2. Match Predictions to Ground Truth ---
    
    for i in range(num_pred):
        p_box = pred_boxes[i]
        max_iou = -1
        best_gt_idx = -1
        
        # Find the best GT match for the current prediction
        for gt_idx in range(num_gt):
            iou = calculate_iou(p_box, gt_boxes[gt_idx])
            
            if iou > max_iou:
                max_iou = iou
                best_gt_idx = gt_idx

        # Check for True Positive condition
        if max_iou >= iou_threshold:
            # Check if this GT box has already been matched
            if not gt_matched[best_gt_idx]:
                true_positive_ious.append(max_iou)
                gt_matched[best_gt_idx] = True # Mark the GT as matched
            # else: This is a duplicate detection (False Positive), ignored for these metrics

    # TP count is the number of GT boxes successfully matched
    true_positives_count = np.sum(gt_matched)
    
    # 1. Overall Recall
    # Recall = TP / (Total GT)
    recall = true_positives_count / num_gt
    
    # 2. Average IoU of TPs
    avg_iou = np.mean(true_positive_ious) if true_positive_ious else 0.0
    
    return recall, avg_iou, true_positives_count
