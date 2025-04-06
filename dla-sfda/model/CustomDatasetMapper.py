"""
File: dla-sfda/model/CustomDatasetMapper.py
Description: CustomDatasetMapper Class
"""

import copy
from typing import List, Any, Dict

import numpy as np
import torch
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
import torchvision.transforms as tv_transforms
from PIL import Image

class CustomDatasetMapper:
    """
    Description:
        A custom dataset mapper for processing input data with weak and strong augmentations.
    Params:
        is_train (bool): Indicates whether the mapper is used for training.
        augmentations (T.AugmentationList): List of weak augmentations to be applied.
        image_format (str): The format of the input images (e.g., "RGB").
        strong_augmentation (Optional[tv_transforms.Compose]): Strong augmentation pipeline.
    """
    def __init__(self, is_train: bool, augmentations: List[Any], image_format: str, strong_augmentation=None):
        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        self.strong_augmentation = strong_augmentation

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        """
        Description:
            Creates an instance of CustomDatasetMapper from a configuration object.
        Returns:
            CustomDatasetMapper: Initialized dataset mapper instance.
        """
        augmentations = utils.build_augmentation(cfg, is_train)
        strong_augmentation = cls.build_strong_augmentation(cfg, is_train)
        return cls(
            is_train=is_train,
            augmentations=augmentations,
            image_format=cfg.INPUT.FORMAT,
            strong_augmentation=strong_augmentation,
        )

    @staticmethod
    def build_strong_augmentation(cfg, is_train):
        if not is_train:
            return None
        strong_aug = tv_transforms.Compose([
            tv_transforms.RandomApply([
                tv_transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            tv_transforms.RandomGrayscale(p=0.2),
            tv_transforms.RandomApply([tv_transforms.GaussianBlur(kernel_size=3)], p=0.5),
        ])
        return strong_aug

    def __call__(self, dataset_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Description:
            Processes a single dataset dictionary.
        Params:
            dataset_dict (Dict[str, Any]): The dataset dictionary containing input data.
        Returns:
            dataset_dict (Dict[str, Any]): Processed dataset dictionary with augmented images and annotations.
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        image_original = image.copy()
        dataset_dict["image_original"] = torch.as_tensor(
            np.ascontiguousarray(image_original.transpose(2, 0, 1))
        )

        # Weak augmentations
        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image_weak_aug = aug_input.image

        # Strong augmentations
        image_pil = Image.fromarray(image_weak_aug.astype("uint8"), "RGB")
        image_strong_aug = np.array(self.strong_augmentation(image_pil))

        # Convert images to tensors
        dataset_dict["image_weak"] = torch.as_tensor(
            np.ascontiguousarray(image_weak_aug.transpose(2, 0, 1))
        )
        dataset_dict["image_strong"] = torch.as_tensor(
            np.ascontiguousarray(image_strong_aug.transpose(2, 0, 1))
        )

        # Use weakly augmented image as the 'image' key
        #dataset_dict["image"] = dataset_dict["image_original"]
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image_weak_aug.transpose(2, 0, 1))
        )

        if "annotations" in dataset_dict:
            # Apply transforms to annotations
            annos = [
                #utils.transform_instance_annotations(
                #    obj, transforms, image.shape[:2]
                #)
                utils.transform_instance_annotations(
                    obj, transforms, image_weak_aug.shape[:2]
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # Convert annotations to Instances
            instances = utils.annotations_to_instances(annos, image.shape[:2])
            dataset_dict["instances"] = instances

        return dataset_dict


