"""
File: dla-sfda/model/student_model.py
Description: Metamodel for the Student Class
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
from detectron2.modeling import META_ARCH_REGISTRY, Backbone, build_backbone, build_roi_heads, build_proposal_generator, detector_postprocess
from torch import nn
from detectron2.config import configurable
from detectron2.structures import ImageList, Instances

from tools.helper_functions import clean_and_get_memory, instances_to_dict


@META_ARCH_REGISTRY.register()
class student_model_dla_sfda(nn.Module):
    """Metamodel for the Student Class"""

    @configurable
    def __init__(self, *, backbone: Backbone, proposal_generator: nn.Module, roi_heads: nn.Module, pixel_mean: Tuple[float], pixel_std: Tuple[float], input_format: Optional[str] = None,):
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (self.pixel_mean.shape == self.pixel_std.shape), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"


    @classmethod
    def from_config(cls, cfg):
        """
        Description:
            Creates an instance of the student model from a configuration object.
        Params:
            cfg: Configuration object containing model parameters.
        Returns:
            Dictionary of parameters for initializing the student model.
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

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], t_results: Optional[List[Instances]] = None, mode: str = "test", aug: str=None):
        """
        Description:
            Forward pass of the model
        Params:
            batched_inputs (List[Dict[str, torch.Tensor]]): Batch of input data
            t_results (Optional[List[Instances]]): Target results for training
            mode (str): "test" or "train"
            aug (Optional[str]): Augmentation type ("strong", "weak", None)
        Returns:
            features, proposals, proposal_losses, results, detector_losses OR inference results
        """

        batched_inputs = [
            {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in d.items()} for d in batched_inputs
        ]
        if t_results is not None:
            t_results = [t.to(self.device) for t in t_results]

        if not self.training and mode == "test":
            return self._inference(batched_inputs, aug=aug)

        images = self._preprocess_image(batched_inputs, aug)
        features = self.backbone(images.tensor)

        if t_results is None:
            return features, None, None, None, None
        else:
            proposals, proposal_losses = self.proposal_generator(images, features, t_results)
            _, detector_losses = self.roi_heads(images, features, proposals, t_results)

            return features, proposals, proposal_losses, detector_losses


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
            return student_model_dla_sfda._postprocess(results, batched_inputs, images.image_sizes)
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
