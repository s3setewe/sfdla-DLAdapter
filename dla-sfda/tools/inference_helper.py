"""
File: dla-sfda/tools/inference_helper.py
Description: Functions for Inference / Testing
"""

import os
from typing import Any

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_setup
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling import build_model
from detectron2.evaluation import inference_on_dataset

from model.CustomCOCOEvaluator import CustomCOCOEvaluator


def test_sfda(cfg: Any, model: Any):
    """
    Description:
        Runs testing on a given model and configuration.
    Params:
        cfg (Any): configuration object
        model (Any): trained model
    Return:
        Results from the inference process, including evaluation metrics.
    """
    assert len(cfg.DATASETS.TEST) == 1
    dataset_name = cfg.DATASETS.TEST[0]
    test_data_loader = build_detection_test_loader(cfg, dataset_name)
    evaluator = CustomCOCOEvaluator(dataset_name, cfg, False, output_dir=str(os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)))
    #evaluator = COCOEvaluator(dataset_name, output_dir=str(os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)))
    results = inference_on_dataset(model, test_data_loader, evaluator)
    return results

def run_inference_with_evaluation(dataset_name: str, annotation_json_path: str, image_root_dln_path: str, repo_source_root: str, model_config_yaml: str, model_path: str):
    """
    Description:
        Registers a dataset, loads a model, and runs inference with evaluation.
    Params:
        dataset_name (str): name of the dataset
        annotation_json_path (str): path to the annotation JSON file.
        image_root_dln_path (str): path to the root directory containing the dataset images.
        repo_source_root (str): root directory of the repository.
        model_config_yaml (str): path to the model configuration YAML file.
        model_path (str): path to the model checkpoint
    Return:
        Results from the inference process, including evaluation metrics.
    """
    if dataset_name in DatasetCatalog:
        DatasetCatalog.remove(dataset_name)
        MetadataCatalog.remove(dataset_name)
    register_coco_instances(dataset_name, {}, annotation_json_path, image_root_dln_path)

    cfg = get_cfg()
    cfg.merge_from_file(f"{repo_source_root}{model_config_yaml}")
    cfg.DATASETS.TEST = dataset_name
    cfg.OUTPUT_DIR = "/workspace"
    cfg.freeze()
    default_setup(cfg, {})

    model = build_model(cfg)
    DetectionCheckpointer(model).load(model_path)
    model.eval()

    test_data_loader = build_detection_test_loader(cfg, dataset_name)
    #evaluator = COCOEvaluator(dataset_name)
    evaluator = CustomCOCOEvaluator(dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR)
    results = inference_on_dataset(model, test_data_loader, evaluator)
    return results
