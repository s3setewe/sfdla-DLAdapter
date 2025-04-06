"""
File: dla-sfda/tools/pseudo_labels.py
Description: Functions for Pseudo Label Processing and dynamic Selection
"""
from typing import List, Dict, Union, Tuple
import numpy as np
import torch
from detectron2.structures import Instances, Boxes

from configs.SFDA_DLA_Configuration_Loader import get_size_category


def process_pseudo_label(proposals_rpn_k: List[Instances], cur_threshold: float) -> List[Instances]:
    """
    Description:
        Processes pseudo labels by filtering proposals based on a confidence threshold
    Params:
        proposals_rpn_k (List[Instances]): list of RPN proposals for an image
        cur_threshold (float): confidence threshold to filter proposals
    Return:
        list_instances (List[Instances]): Filtered list of instances containing bounding boxes, classes, and scores
    """
    list_instances = []
    for proposal_bbox_inst in proposals_rpn_k:
        valid_map = proposal_bbox_inst.scores > cur_threshold

        # create instances containing boxes and gt_classes
        image_shape = proposal_bbox_inst.image_size
        new_proposal_inst = Instances(image_shape)

        # create box
        new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
        new_boxes = Boxes(new_bbox_loc)

        # add boxes to instances
        new_proposal_inst.gt_boxes = new_boxes
        new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
        new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        list_instances.append(new_proposal_inst)
    return list_instances

def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """
    Description:
        Calculates the Intersection over Union (IoU) of two bounding boxes
    Params:
        boxA (List[float]): first bounding box [x1, y1, x2, y2]
        boxB (List[float]): second bounding box [x1, y1, x2, y2]
    Return:
        float: IoU value
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    intersection_area = max(0, xB - xA) * max(0, yB - yA)

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if (boxA_area + boxB_area - intersection_area) == 0:
        return 0.0

    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)
    return iou


def is_fully_contained(boxA: List[float], boxB: List[float]) -> bool:
    """
    Description:
        checks if one bounding box (boxB) is fully contained within another (boxA)
    Params:
        boxA (List[float]): outer bounding box [x1, y1, x2, y2]
        boxB (List[float]): inner bounding box [x1, y1, x2, y2]
    Return:
        bool: True if boxB is fully contained within boxA, False otherwise
    """
    return (boxA[0] <= boxB[0] and boxA[1] <= boxB[1] and
            boxA[2] >= boxB[2] and boxA[3] >= boxB[3])


def my_selection_pseudo_labels_process(
    dynamic_teacher_pseudo_proposals_dict: Dict[str, Union[List[List[float]], List[int], List[float]]],
    static_teacher_pseudo_proposals_dict: Dict[str, Union[List[List[float]], List[int], List[float]]],
    categories: Dict[int, Dict[str, float]],
    image_height: int,
    image_width: int,
    device: torch.device,
    similar_classes: List[int],
    sigma: float,
    similarity_threshold: float,
    alpha: float,
    beta: float,
    boost_factor: float,
    no_match_found: float,
    non_maximum_threshold: float,
    matching_weight: float,
    num_classes: int,
    redist: float
) -> Tuple[Dict[str, Union[List[List[float]], List[int], List[float], List[List[float]]]], List[Instances], Dict[str, Union[List[List[float]], List[int], List[float], List[List[float]]]]]:
    """
    Description:
        Processes pseudo labels from dynamic and static teachers, combining them based on similarity metrics.
    Params:
        dynamic_teacher_pseudo_proposals_dict: Dynamic teacher proposals with keys 'bboxes', 'classes', 'scores', and optional 'logits'
        static_teacher_pseudo_proposals_dict: Static teacher proposals with keys 'bboxes', 'classes', 'scores', and optional 'logits'
        categories: Class-specific thresholds based on size categories
        image_height (int): height of the image
        image_width (int): width of the image
        device (torch.device): PyTorch device
        similar_classes (list): class_indices_which are similar
        sigma (float)
        similarity_threshold (float)
        alpha (float)
        beta (float)
        boost_factor (float)
        matching_weight (float)
        num_classes (int)
        redist (float)
    Return:
        Tuple: Dictionary containing selected pseudo labels, corresponding instances
    """

    def gaussian_weight(dt_bbox, st_bbox, sigma):
        dt_cx = (dt_bbox[0] + dt_bbox[2]) / 2.0
        dt_cy = (dt_bbox[1] + dt_bbox[3]) / 2.0
        st_cx = (st_bbox[0] + st_bbox[2]) / 2.0
        st_cy = (st_bbox[1] + st_bbox[3]) / 2.0
        dx = dt_cx - st_cx
        dy = dt_cy - st_cy
        distance = (dx ** 2 + dy ** 2) ** 0.5
        distance_tensor = torch.tensor(distance, dtype=torch.float32)
        return torch.exp(- (distance_tensor ** 2) / (2 * sigma ** 2)).item()

    def class_distribution_similarity(dt_logits, st_logits, alpha):
        if (not dt_logits) or (not st_logits):
            return 0.0
        dt_probs = torch.softmax(torch.tensor(dt_logits, device=device), dim=0)
        st_probs = torch.softmax(torch.tensor(st_logits, device=device), dim=0)
        kl_divergence = torch.nn.functional.kl_div(dt_probs.log(), st_probs, reduction='batchmean')
        cosine_sim = torch.nn.functional.cosine_similarity(dt_probs.unsqueeze(0), st_probs.unsqueeze(0)).mean().item()
        return alpha * (1 - kl_divergence.item()) + (1-alpha) * cosine_sim

    def combined_similarity(iou, class_sim, beta):
        return beta * iou + (1-beta) * class_sim

    def calculate_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        intersection_area = max(0, xB - xA) * max(0, yB - yA)
        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = intersection_area / float(boxA_area + boxB_area - intersection_area + 0.00001)
        return iou

    dynamic_static_consens_mechanism = {
        'bboxes': [],
        'classes': [],
        'scores': [],
        'dt_logits': [],
        'st_logits': []
    }

    for dt_bbox, dt_class_id, dt_score, dt_logits in zip(
        dynamic_teacher_pseudo_proposals_dict['bboxes'],
        dynamic_teacher_pseudo_proposals_dict['classes'],
        dynamic_teacher_pseudo_proposals_dict['scores'],
        dynamic_teacher_pseudo_proposals_dict.get('logits', [])
    ):
        found_match = False
        best_combined_score = -1.0
        best_merged_bbox = None
        best_class = dt_class_id
        best_score = dt_score * no_match_found
        best_dt_logits = dt_logits
        best_st_logits = None

        for st_bbox, st_class_id, st_score, st_logits in zip(
            static_teacher_pseudo_proposals_dict['bboxes'],
            static_teacher_pseudo_proposals_dict['classes'],
            static_teacher_pseudo_proposals_dict['scores'],
            static_teacher_pseudo_proposals_dict.get('logits', [])
        ):
            iou = calculate_iou(dt_bbox, st_bbox)
            class_sim = class_distribution_similarity(dt_logits, st_logits, alpha)
            sim = combined_similarity(iou, class_sim, beta)

            if dt_class_id == st_class_id and sim >= similarity_threshold:
                w = gaussian_weight(dt_bbox, st_bbox, sigma=sigma)
                merged_bbox = [
                    w * dt_bbox[0] + (1 - w) * st_bbox[0],
                    w * dt_bbox[1] + (1 - w) * st_bbox[1],
                    w * dt_bbox[2] + (1 - w) * st_bbox[2],
                    w * dt_bbox[3] + (1 - w) * st_bbox[3],
                ]
                merged_score = w * dt_score + (1 - w) * st_score

                if merged_score > best_combined_score:
                    best_combined_score = merged_score
                    best_merged_bbox = merged_bbox
                    best_class = dt_class_id
                    best_score = merged_score
                    best_st_logits = st_logits
                found_match = True

        if found_match and best_merged_bbox is not None:
            dynamic_static_consens_mechanism['bboxes'].append(best_merged_bbox)
            dynamic_static_consens_mechanism['classes'].append(best_class)
            dynamic_static_consens_mechanism['scores'].append(best_score)
            dynamic_static_consens_mechanism['dt_logits'].append(best_dt_logits)
            dynamic_static_consens_mechanism['st_logits'].append(best_st_logits)
        else:
            dynamic_static_consens_mechanism['bboxes'].append(dt_bbox)
            dynamic_static_consens_mechanism['classes'].append(dt_class_id)
            dynamic_static_consens_mechanism['scores'].append(dt_score * no_match_found)
            dynamic_static_consens_mechanism['dt_logits'].append(dt_logits)
            dynamic_static_consens_mechanism['st_logits'].append(None)

    my_selection_pseudo_labels_dict = {
        'bboxes': [],
        'classes': [],
        'scores': [],
        'soft_distribution': [],
        'type': "My Selection"
    }

    def non_max_suppression(bboxes, scores, classes, iou_threshold):
        idxs = np.argsort(scores)[::-1]
        selected_bboxes = []
        selected_scores = []
        selected_classes = []

        def calc_iou(a, b):
            xA = max(a[0], b[0])
            yA = max(a[1], b[1])
            xB = min(a[2], b[2])
            yB = min(a[3], b[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (a[2] - a[0]) * (a[3] - a[1])
            boxBArea = (b[2] - b[0]) * (b[3] - b[1])
            return interArea / float(boxAArea + boxBArea - interArea + 0.00001)

        while len(idxs) > 0:
            current_idx = idxs[0]
            selected_bboxes.append(bboxes[current_idx])
            selected_scores.append(scores[current_idx])
            selected_classes.append(classes[current_idx])

            remove_idxs = [0]
            for i in range(1, len(idxs)):
                iou = calc_iou(bboxes[current_idx], bboxes[idxs[i]])
                if iou > iou_threshold or is_fully_contained(bboxes[current_idx], bboxes[idxs[i]]):
                    remove_idxs.append(i)
            idxs = np.delete(idxs, remove_idxs)

        return selected_bboxes, selected_scores, selected_classes


    for bbox, cls_id, score, dt_logits, st_logits in zip(
        dynamic_static_consens_mechanism['bboxes'],
        dynamic_static_consens_mechanism['classes'],
        dynamic_static_consens_mechanism['scores'],
        dynamic_static_consens_mechanism['dt_logits'],
        dynamic_static_consens_mechanism['st_logits']
    ):
        dt_probs = torch.softmax(torch.tensor(dt_logits, device=device), dim=0) if dt_logits else None
        st_probs = torch.softmax(torch.tensor(st_logits, device=device), dim=0) if st_logits else None

        if dt_probs is not None and st_probs is not None:

            combined_probs = matching_weight * dt_probs + (1 - matching_weight) * st_probs
        elif dt_probs is not None:
            combined_probs = dt_probs
        elif st_probs is not None:
            combined_probs = st_probs
        else:
            combined_probs = torch.zeros(num_classes, device=device)
            combined_probs[cls_id] = 1.0

        similar_mean = combined_probs[similar_classes].mean()
        combined_probs[similar_classes] = ((1-redist) * combined_probs[similar_classes] + redist * similar_mean)
        combined_probs = combined_probs / combined_probs.sum()

        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        area_size_category = get_size_category(area=area)
        threshold = categories[cls_id][area_size_category] if area_size_category in categories[cls_id] else 0.5

        if score > threshold:
            combined_probs[cls_id] = combined_probs[cls_id] * boost_factor
            combined_probs = combined_probs / combined_probs.sum()
            my_selection_pseudo_labels_dict['bboxes'].append(bbox)
            my_selection_pseudo_labels_dict['classes'].append(cls_id)
            my_selection_pseudo_labels_dict['scores'].append(score)
            my_selection_pseudo_labels_dict['soft_distribution'].append(combined_probs.cpu().numpy().tolist())


    if len(my_selection_pseudo_labels_dict['bboxes']) > 0:
        nms_bboxes, nms_scores, nms_classes = non_max_suppression(
            my_selection_pseudo_labels_dict['bboxes'],
            my_selection_pseudo_labels_dict['scores'],
            my_selection_pseudo_labels_dict['classes'],
            non_maximum_threshold
        )

        final_bboxes = []
        final_scores = []
        final_classes = []
        final_distributions = []
        for nb in nms_bboxes:
            for orig_idx, ob in enumerate(my_selection_pseudo_labels_dict['bboxes']):
                if np.allclose(nb, ob, atol=1e-5):
                    final_bboxes.append(ob)
                    final_scores.append(my_selection_pseudo_labels_dict['scores'][orig_idx])
                    final_classes.append(my_selection_pseudo_labels_dict['classes'][orig_idx])
                    final_distributions.append(my_selection_pseudo_labels_dict['soft_distribution'][orig_idx])
                    break
        my_selection_pseudo_labels_dict['bboxes'] = final_bboxes
        my_selection_pseudo_labels_dict['scores'] = final_scores
        my_selection_pseudo_labels_dict['classes'] = final_classes
        my_selection_pseudo_labels_dict['soft_distribution'] = final_distributions

    selected_teacher_pseudo_results = [Instances((image_height, image_width))]
    if len(my_selection_pseudo_labels_dict['bboxes']) > 0:
        selected_teacher_pseudo_results[0].gt_boxes = Boxes(torch.tensor(my_selection_pseudo_labels_dict['bboxes'], device=device))
        selected_teacher_pseudo_results[0].scores = torch.tensor(my_selection_pseudo_labels_dict['scores'], device=device)
        selected_teacher_pseudo_results[0].gt_classes = torch.tensor(my_selection_pseudo_labels_dict['classes'], device=device)

        soft_distributions = torch.tensor(my_selection_pseudo_labels_dict['soft_distribution'], device=device)
        selected_teacher_pseudo_results[0].soft_cls_distribution = soft_distributions
    else:
        selected_teacher_pseudo_results[0].gt_boxes = Boxes(torch.empty((0,4), device=device))
        selected_teacher_pseudo_results[0].scores = torch.empty((0,), device=device)
        selected_teacher_pseudo_results[0].gt_classes = torch.empty((0,), device=device, dtype=torch.long)
        selected_teacher_pseudo_results[0].soft_cls_distribution = torch.empty((0,num_classes), device=device)

    return my_selection_pseudo_labels_dict, selected_teacher_pseudo_results, dynamic_static_consens_mechanism
