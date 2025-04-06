"""
File: dla-sfda/model/CustomCOCOEvaluator.py
Description: CustomCOCOEvaluator Class
"""

import itertools

import numpy as np
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import create_small_table
from tabulate import tabulate


class CustomCOCOEvaluator(COCOEvaluator):
    """
    Description:
        This class contains a custom evaluator for COCO that performs evaluations for mAP 0.5 and 0.95 (desired_iou_thresholds).
        It extends COCOEvaluator.
    """
    # def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
    #     """
    #     Override the method to extract specific IoU thresholds (e.g., IoU=0.5 and IoU=0.95 for AP50 and AP95).
    #     """
    #     if coco_eval is None:
    #         return {}
    #
    #     # Original stats
    #     metrics = {
    #         "AP": coco_eval.stats[0],  # mAP@[0.5:0.95]
    #         "AP50": coco_eval.stats[1],  # mAP@50
    #         "AP75": coco_eval.stats[2],  # mAP@75
    #         "APs": coco_eval.stats[3],  # AP for small objects
    #         "APm": coco_eval.stats[4],  # AP for medium objects
    #         "APl": coco_eval.stats[5],  # AP for large objects
    #     }
    #
    #     # Define the IoU thresholds you want to extract
    #     desired_iou_thresholds = [0.5, 0.95]
    #
    #     # Store original iouThrs to restore later
    #     original_iouThrs = copy.deepcopy(coco_eval.params.iouThrs)
    #
    #     for iou_threshold in desired_iou_thresholds:
    #         # Set the IoU threshold to the current value
    #         coco_eval.params.iouThrs = [iou_threshold]
    #
    #         # Reset evaluation
    #         coco_eval.eval = {}
    #         coco_eval.stats = []
    #
    #         # Perform evaluation
    #         coco_eval.evaluate()
    #         coco_eval.accumulate()
    #         coco_eval.summarize()
    #
    #         # Extract per-category AP at the current IoU threshold
    #         if class_names is not None:
    #             per_category_results = {}
    #             precisions = coco_eval.eval.get("precision")
    #
    #             if precisions is not None:
    #                 # precision has shape [T, R, K, A, M]
    #                 # T=1 since we set iouThrs to a single value
    #                 # Extract the first (and only) IoU threshold
    #                 precision = precisions[0, :, :, 0, -1]  # [R, K]
    #
    #                 # Iterate over each class
    #                 for idx, name in enumerate(class_names):
    #                     # Extract valid precision values (exclude -1)
    #                     class_precision = precision[:, idx]
    #                     valid_precisions = class_precision[class_precision > -1]
    #                     ap = valid_precisions.mean() if valid_precisions.size > 0 else float("nan")
    #                     per_category_results[name] = ap
    #
    #             # Add the per-category AP to metrics with appropriate key
    #             metrics[f"per_category_AP{int(iou_threshold * 100)}"] = per_category_results
    #
    #     # Restore the original IoU thresholds
    #     coco_eval.params.iouThrs = original_iouThrs
    #
    #     return metrics

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Override the method to extract specific IoU thresholds (e.g., IoU=0.5 and IoU=0.95 for AP50 and AP95).

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """
        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results

        # Compute per-category AP for AP, AP50, AP75, and AP95
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = {iou_thresh: [] for iou_thresh in [0.5, 0.75, 0.95]}
        iou_thresh_indices = {0.5: 0, 0.75: 5, 0.95: 9}  # COCOeval thresholds for AP50, AP75, and AP95

        for idx, name in enumerate(class_names):
            for iou_thresh, iou_idx in iou_thresh_indices.items():
                precision = precisions[iou_idx, :, idx, 0, -1]
                precision = precision[precision > -1]
                ap = np.mean(precision) if precision.size else float("nan")
                results_per_category[iou_thresh].append(("{}".format(name), float(ap * 100)))

        # Tabulate results
        for iou_thresh, per_category_results in results_per_category.items():
            N_COLS = min(6, len(per_category_results) * 2)
            results_flatten = list(itertools.chain(*per_category_results))
            results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
            table = tabulate(
                results_2d,
                tablefmt="pipe",
                floatfmt=".3f",
                headers=["category", f"AP@{iou_thresh:.0%}"] * (N_COLS // 2),
                numalign="left",
            )
            self._logger.info(f"Per-category {iou_type} AP@{iou_thresh:.0%}: \n" + table)

        # Update results dictionary
        for iou_thresh, per_category_results in results_per_category.items():
            results.update({f"AP@{iou_thresh:.0%}-" + name: ap for name, ap in per_category_results})

        return results

