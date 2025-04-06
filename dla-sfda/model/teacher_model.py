"""
File: dla-sfda/model/teacher_model.py
Description: Metamodel for the Teacher Class
"""

from typing import Dict, List, Optional, Tuple
import torch
from detectron2.modeling import META_ARCH_REGISTRY, Backbone, build_proposal_generator, build_roi_heads, build_backbone, detector_postprocess
from torch import nn

from detectron2.config import configurable
from detectron2.structures import ImageList, Instances


@META_ARCH_REGISTRY.register()
class teacher_model_dla_sfda(nn.Module):
    """Metamodel for the Teacher Class"""

    @configurable
    def __init__(self, *, backbone: Backbone, proposal_generator: nn.Module, roi_heads: nn.Module, pixel_mean: Tuple[float], pixel_std: Tuple[float], input_format: Optional[str] = None):

        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.num_classes = roi_heads.num_classes

        self.input_format = input_format

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (self.pixel_mean.shape == self.pixel_std.shape), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        """
        Description:
            Creates an instance of the teacher model from a configuration object.
        Params:
            cfg: Configuration object containing model parameters.
        Returns:
            Dictionary of parameters for initializing the teacher model.
        """
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        """Returns the device on which the model is located (torch.device)"""
        return self.pixel_mean.device

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], mode="test", aug: str=None, noise_std: float = 0.0, class_noise_prob: float = 0.0, num_extra_boxes: int = 0):
        """
        Description:
            Forward pass of the model
        Params:
            batched_inputs (List[Dict[str, torch.Tensor]]): Batch of input data
            mode (str): "test" or "train"
            aug (Optional[str]): Augmentation type ("strong", "weak", None)
            noise_std (float): noise value for bboxes
            class_noise_prob (float): prob for class change
            num_extra_boxes (int): number of random synthetic boxes
        Returns:
            features, proposals, proposal_losses, results, detector_losses OR inference results.
        """

        if not self.training and mode == "test":
            return self._inference(batched_inputs, aug=aug)

        images = self._preprocess_image(batched_inputs, aug=aug)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        features = self.backbone(images.tensor)

        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        results, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        if noise_std > 0.0 or class_noise_prob > 0.0:
            results = self._add_noise_to_bboxes_and_classes(results, noise_std, class_noise_prob, max_width=images.image_sizes[0][0], max_height=images.image_sizes[0][1])

        if num_extra_boxes > 0:
            results = self._add_synthetic_bboxes(results, num_extra_boxes=num_extra_boxes, max_width=images.image_sizes[0][0], max_height=images.image_sizes[0][1])

        return features, proposals, proposal_losses, results, detector_losses

    def _inference(self, batched_inputs: List[Dict[str, torch.Tensor]], aug: str, detected_instances: Optional[List[Instances]] = None, do_postprocess: bool = True):
        """Performs inference on the input data."""
        assert not self.training

        images = self._preprocess_image(batched_inputs, aug)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return teacher_model_dla_sfda._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def _preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]], aug: str):
        """Preprocesses input images for the model: Augmentations and Pixel Normalization"""
        if aug == "strong":
            images = [x["image_strong"].to(self.device) for x in batched_inputs]
        elif aug == "weak":
            images = [x["image_weak"].to(self.device) for x in batched_inputs]
        else:
            images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """Postprocesses model predictions to match the input image dimensions."""
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(instances, batched_inputs, image_sizes):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    # def _add_noise_to_bboxes(self, results: List[Instances], noise_std: float):
    #     """
    #     Add Noise to Bbox Predictions
    #     results: List of Instances
    #     noise_std: value for noise
    #     """
    #     for instances in results:
    #         if instances.has("pred_boxes"):
    #             boxes = instances.pred_boxes.tensor
    #             noise = torch.randn_like(boxes) * noise_std
    #             boxes += noise
    #
    #             boxes[:, 0].clamp_(min=0)
    #             boxes[:, 1].clamp_(min=0)
    #             boxes[:, 2].clamp_(min=boxes[:, 0] + 1)
    #             boxes[:, 3].clamp_(min=boxes[:, 1] + 1)
    #
    #             instances.pred_boxes.tensor = boxes
    #
    #     return results

    def _add_noise_to_bboxes_and_classes(self, results: List[Instances], noise_std: float, class_noise_prob: float, max_width: int, max_height: int):
        """
        Add Noise to Bbox Predictions
        results: List of Instances
        noise_std: value for noise
        class_noise_prob: probability for changing a class
        num_classes: number of model classes
        max_width: max width of images
        max_height: max_height of images

        returns list with noise bboxes
        """
        for instances in results:
            if instances.has("pred_boxes"):
                boxes = instances.pred_boxes.tensor
                noise = torch.randn_like(boxes) * noise_std
                boxes += noise

                boxes[:, 0].clamp_(min=0, max=max_width)
                boxes[:, 1].clamp_(min=0, max=max_height)
                boxes[:, 2].clamp_(min=(boxes[:, 0] + 1), max=torch.tensor(max_width, device=boxes.device))
                boxes[:, 3].clamp_(min=(boxes[:, 1] + 1), max=torch.tensor(max_height, device=boxes.device))
                instances.pred_boxes.tensor = boxes

            if instances.has("pred_classes"):
                mask = torch.rand_like(instances.pred_classes.float()) < class_noise_prob
                random_classes = torch.randint(0, self.num_classes, instances.pred_classes.shape, device=instances.pred_classes.device)
                instances.pred_classes[mask] = random_classes[mask]

        return results

    def _add_synthetic_bboxes(self, results: List[Instances], num_extra_boxes, max_width: int, max_height: int):
        """
        Add synthetic bounding boxes
        num_extra_boxes: number of extra boxes
        max_width: max width of images
        max_height: max_height of images
        num_classes: number of classes

        returns list with synthetic boxes
        """
        for instances in results:
            x1 = torch.randint(0, max_width // 2, (num_extra_boxes,), device=instances.pred_boxes.device)
            y1 = torch.randint(0, max_height // 2, (num_extra_boxes,), device=instances.pred_boxes.device)
            x2 = x1 + torch.randint(20, max_width // 3, (num_extra_boxes,), device=instances.pred_boxes.device)
            y2 = y1 + torch.randint(20, max_height // 3, (num_extra_boxes,), device=instances.pred_boxes.device)

            x2.clamp_(max=max_width)
            y2.clamp_(max=max_height)

            new_boxes = torch.stack([x1, y1, x2, y2], dim=1)
            new_scores = torch.rand((num_extra_boxes,), device=instances.pred_boxes.device)
            new_classes = torch.randint(0, self.num_classes, (num_extra_boxes,), device=instances.pred_classes.device)

            instances.pred_boxes.tensor = torch.cat([instances.pred_boxes.tensor, new_boxes], dim=0)
            instances.scores = torch.cat([instances.scores, new_scores], dim=0)
            instances.pred_classes = torch.cat([instances.pred_classes, new_classes], dim=0)

        return results
