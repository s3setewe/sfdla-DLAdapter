"""
File: dla-sfda/tools/model_helper.py
Description: Model Loading and Updating
"""
from typing import Any, Tuple, Dict

import torch
from collections import OrderedDict
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import detectron2.utils.comm as comm

def load_models(args: Any, cfg: Any) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    """
    Description:
        Load the student and teacher models based on the configuration.
    Params:
        args (Any): arguments containing configurations
        cfg (Any): detectron2 configuration
    Return:
        A tuple containing the student model, static teacher model, and dynamic teacher model.
    """
    cfg.defrost()
    cfg.MODEL.META_ARCHITECTURE = "student_model_dla_sfda"
    cfg.freeze()
    student_model = build_model(cfg)
    DetectionCheckpointer(student_model, save_dir=cfg.OUTPUT_DIR).load(args.configuration['model_dir'])

    cfg.defrost()
    cfg.MODEL.META_ARCHITECTURE = "teacher_model_dla_sfda"
    cfg.freeze()
    static_teacher_model = build_model(cfg)
    DetectionCheckpointer(static_teacher_model, save_dir=cfg.OUTPUT_DIR).load(args.configuration['model_dir'])

    cfg.defrost()
    cfg.MODEL.META_ARCHITECTURE = "teacher_model_dla_sfda"
    cfg.freeze()
    dynamic_teacher_model = build_model(cfg)
    DetectionCheckpointer(dynamic_teacher_model, save_dir=cfg.OUTPUT_DIR).load(args.configuration['model_dir'])

    return student_model, static_teacher_model, dynamic_teacher_model

@torch.no_grad()
def update_teacher_model(model_student: torch.nn.Module, model_teacher: torch.nn.Module, keep_rate: float) -> Dict[str, torch.Tensor]:
    """
    Description:
        Update the teacher models parameters using the student models parameters with an exponential moving average (EMA)
    Params:
        model_student (torch.nn.Module): student model
        model_teacher (torch.nn.Module): teacher model
        keep_rate (float): rate at which the teacher model retains its own parameters.
    Return:
        updated state dictionary for the teacher model.
    Raises:
        Exception: If a key in the teacher models state dict is not found in the student models state dict.
    """
    if comm.get_world_size() > 1:
        student_model_dict = {key[7:]: value for key, value in model_student.state_dict().items()}  # 7 to cut module.
    else:
        student_model_dict = model_student.state_dict()

    new_teacher_dict = OrderedDict()
    teacher_device = next(model_teacher.parameters()).device
    for key, value in model_teacher.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (student_model_dict[key].to(teacher_device) * (1 - keep_rate) + value * keep_rate)
        else:
            raise Exception("{} is not found in student model".format(key))

    return new_teacher_dict