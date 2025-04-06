"""
File: dla-sfda/tools/helper_functions.py
Description: Simple helper functions
"""
import gc
import random
import shutil
import json
from pathlib import Path
import os
import logging
from typing import Dict, List, Union, Optional, Any

import numpy as np
import psutil
import torch


def get_logger(logger_file_path: str) -> logging.Logger:
    """Initialize and return a logger for logging messages."""
    if os.path.exists(logger_file_path):
        os.remove(logger_file_path)
    if os.path.exists("/workspace/log.txt"):
        os.remove("/workspace/log.txt")
    logger = logging.getLogger("detectron2")
    return logger

def make_visualization_path(visualization_path: str) -> None:
    """Create or reset the directory for visualizations."""
    directory = Path(visualization_path)
    if directory.exists() and directory.is_dir():
        shutil.rmtree(directory)
    os.makedirs(visualization_path, exist_ok=True)

def set_random(seed: int=0) -> None:
    """Set the random seed for reproducibility: torch, random, numpy"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def save_json(path: str, json_dict_list: Any) -> None:
    """Writes the provided dictionary list to a JSON file at the specified path."""
    with open(path, "w") as file:
        json.dump(json_dict_list, file, indent=4)
    print(f"Save to {path}")

def clean_and_get_memory(logger: logging.Logger = None, log=False, clean=False):
    """
    Description:
        Cleans GPU memory and logs system memory statistics if specified.
    Params:
        logger (logging.Logger): A logger instance
        log (bool): If True, logs memory statistics (before and/or after cleanup).
        clean (bool): If True, clears GPU cache and runs garbage collection.
    """
    memory_dict = {}
    if logger and log:
        memory_dict_before =  {
            "Used memory": f"{(psutil.virtual_memory()).used / (1024**3):.2f} GB",
            "Allocated GPU memory": f"{torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB",
            "Cached GPU memory": f"{torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB"
        }
        memory_dict['memory_before'] = memory_dict_before
    if clean:
        torch.cuda.empty_cache()
        gc.collect()
    if logger and log and clean:
        memory_dict_after = {
            "Used memory": f"{(psutil.virtual_memory()).used / (1024 ** 3):.2f} GB",
            "Allocated GPU memory": f"{torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB",
            "Cached GPU memory": f"{torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB"
        }
        memory_dict['memory_after'] = memory_dict_after
    if logger and log:
        logger.info(memory_dict)


def instances_to_dict(instance_results, logits: Optional[torch.Tensor] = None, type: Optional[str] = None) -> Dict[str, Union[List, str]]:
    """
    Description:
        Convert instance results to a dictionary and extracts bounding boxes, classes, scores, logits and type information from instance results.
    Return:
        Dictionary containing extracted information.
    """
    labels_dict = {
        'bboxes': instance_results[0].gt_boxes.tensor.cpu().numpy().tolist(),
        'classes': instance_results[0].gt_classes.cpu().numpy().tolist(),
        'scores': instance_results[0].scores.cpu().numpy().tolist(),
        'logits': None if logits is None else logits.cpu().numpy().tolist(),
        'type': type
    }
    return labels_dict

