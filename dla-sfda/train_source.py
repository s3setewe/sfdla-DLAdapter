"""
File: dla-sfda/train_source.py
Description: Fully Supervised Training of the Source Model
"""

import argparse
import gc
import os
import time
from typing import Any

import torch
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_train_loader
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.engine import default_setup, launch, DefaultTrainer, AMPTrainer

from model.CustomCOCOEvaluator import CustomCOCOEvaluator
from model.CustomDatasetMapper import CustomDatasetMapper
from configs.SourceModelConfig import SourceModelConfig
from tools.helper_functions import get_logger


class CustomTrainer(DefaultTrainer):
    """A custom trainer with custom dataset mapper for fully supervised training of the source model"""
    def __init__(self, cfg: Any):
        super().__init__(cfg)
        self.device = torch.device(cfg.MODEL.DEVICE)

    @classmethod
    def build_train_loader(cls, cfg: Any) -> Any:
        """builds the training data loader using a custom dataset mapper"""
        mapper = CustomDatasetMapper.from_config(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    def run_step(self) -> None:
        """Executes one training step. Fetches data, computes loss, and updates the model"""
        torch.cuda.empty_cache()
        gc.collect()
        assert self.model.training, "[CustomTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        images = [x["image"].to(self.device) for x in data]
        instances = [x["instances"].to(self.device) for x in data]

        batched_inputs = [{"image": img, "instances": inst} for img, inst in zip(images, instances)]

        loss_dict = self.model(batched_inputs)
        losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        self.storage.put_scalar("data_time", data_time)


def setup(args: argparse.Namespace) -> Any:
    """
    Description:
        sets up the detectron2 configuration
    Params:
        args (argparse.Namespace): command-line arguments
    Return:
        detectron2 configuration object
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.configuration['config_file'])
    cfg.freeze()
    default_setup(cfg, args)

    return cfg

def build_evaluator(cfg: Any, dataset_name: str) -> Any:
    """builds a COCO evaluator for evaluation"""
    #evaluator = COCOEvaluator(dataset_name, cfg, True, os.path.join(cfg.OUTPUT_DIR, "inference"))
    evaluator = CustomCOCOEvaluator(dataset_name, cfg, True, output_dir=os.path.join(cfg.OUTPUT_DIR, "inference"))
    return evaluator


def main(args: argparse.Namespace) -> None:
    """
    Description:
         main function for training the source model
    Params:
        args (argparse.Namespace): The command-line arguments
    """

    configuration = args.configuration
    register_coco_instances(
        configuration["train_dataset_name"], {},
        configuration["train_dataset_annotation_path"],
        configuration["train_dataset_image_folder_path"]
    )
    register_coco_instances(
        configuration["val_dataset_name"], {},
        configuration["val_dataset_annotation_path"],
        configuration["val_dataset_image_folder_path"]
    )

    cfg = setup(args)
    model = build_model(cfg)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    model.eval()

    CustomTrainer.build_evaluator = build_evaluator
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()


if __name__ == "__main__":
    """Entry point for the script. Parses arguments and launches the training"""
    name = "dln_law"
    logger = get_logger(f"/workspace/log_{name}.txt")

    configuration = SourceModelConfig(name).get_configuration_dict()
    print(configuration)

    args = argparse.Namespace(
        dist_url='tcp://127.0.0.1:54160',
        eval_only=False,
        machine_rank=0,
        num_gpus=4,
        num_machines=1,
        opts=[],
        resume=False,
        logger=logger,
        configuration=configuration
    )
    logger.info(args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
