"""
File: dla-sfda/train_sfda.py
Description: SF DLA Training Script
"""

import argparse
import os
from typing import Any
import torch
import torch.multiprocessing

from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import launch, default_setup
from detectron2.solver import build_lr_scheduler
from detectron2.utils.events import EventStorage
from detectron2.data import build_detection_train_loader, DatasetCatalog

from model.CustomDatasetMapper import CustomDatasetMapper
from configs.SFDA_DLA_Configuration_Loader import SFDA_DLA_Configuration_Loader
from tools.helper_functions import get_logger, set_random, clean_and_get_memory, instances_to_dict
from tools.loss_function import compute_contrastive_loss, soft_label_kl_distillation_loss, entropy_loss, feature_distillation_loss
from tools.model_helper import load_models, update_teacher_model
from tools.pseudo_labels import process_pseudo_label, my_selection_pseudo_labels_process
from tools.train_helper import build_custom_optimizer, SortedSampler, compute_entropy_metrics
from tools.inference_helper import test_sfda

from model.student_model import student_model_dla_sfda
from model.teacher_model import teacher_model_dla_sfda

torch.multiprocessing.set_sharing_strategy('file_system')


def train_sfda(args, cfg, student_model, static_teacher_model, dynamic_teacher_model):
    """
    Description:
        train the student model with sfda
    Params:
        args: parsed command-line arguments containing configuration and settings
        cfg: detectron2 configuration object
        student_model: student model for training
        static_teacher_model: static teacher model
        dynamic_teacher_model: dynamic teacher model
    """

    # Set model training/evaluation modes
    student_model.train()
    static_teacher_model.eval()
    dynamic_teacher_model.eval()

    # Configure optimizer and scheduler and initialize checkpointing and random seeds
    optimizer = build_custom_optimizer(cfg, student_model)
    optimizer.zero_grad()
    scheduler = build_lr_scheduler(cfg, optimizer)
    checkpointer = DetectionCheckpointer(student_model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
    set_random(seed=args.seed)

    # Prepare data loader for training
    data_loader = build_detection_train_loader(cfg, mapper=CustomDatasetMapper.from_config(cfg, is_train=True), num_workers=0)
    len_data_loader = len(data_loader.dataset.dataset.dataset)
    start_iter, max_iter = 0, len_data_loader
    max_sf_da_iter = args.total_epochs*max_iter
    num_images = len(data_loader.dataset.dataset.dataset)
    periodic_checkpointer = PeriodicCheckpointer(checkpointer, len_data_loader, max_iter=max_sf_da_iter)

    # Use EventStorage to manage metrics and events
    with EventStorage(start_iter) as storage:
        for epoch in range(1, args.total_epochs+1):
            logger.info(f"Start Epoch {epoch}")

            # Update data loader if a specific sampling strategy is selected (e.g., entropy-based sampling)
            if args.use_entropy_samling:
                metrics = compute_entropy_metrics(cfg, dynamic_teacher_model)
                dataset = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
                sampler = SortedSampler(dataset, metrics, sort_ascending=True)
                data_loader = build_detection_train_loader(
                    cfg,
                    mapper=CustomDatasetMapper.from_config(cfg, is_train=True),
                    num_workers=0,
                    sampler=sampler
                )
            else:
                data_loader = build_detection_train_loader(cfg, mapper=CustomDatasetMapper.from_config(cfg, is_train=True), num_workers=0)

            # Set model modes for the current epoch
            student_model.train()
            static_teacher_model.eval()
            dynamic_teacher_model.eval()

            for data, iteration in zip(data_loader, range(0, num_images)):
                # Periodic memory cleanup and evaluation
                if iteration%args.test_periods==0:
                    clean_and_get_memory(logger=logger, clean=True, log=True)
                    student_model.eval()
                    logger.info(f"Student model testing: Epoch {epoch}, Iteration {iteration}")
                    results = test_sfda(cfg, student_model)
                    logger.info(results)
                    student_model.train()
                    clean_and_get_memory(logger=logger, clean=True, log=True)

                storage.iter = iteration # Update EventStorage iteration counter

                # Dynamic teacher model inference (with weak augmentation)
                with torch.no_grad():
                   dynamic_teacher_features, dynamic_teacher_proposals, _, dynamic_teacher_results, _ = dynamic_teacher_model(data, mode="train", aug="weak", noise_std=args.noise_dynamic_teacher, class_noise_prob=args.class_noise_prob_dynamic_teacher, num_extra_boxes=args.num_extra_boxes_dynamic_teacher)
                   dt_box_features = dynamic_teacher_model.roi_heads._shared_roi_transform([dynamic_teacher_features['res4']], [dynamic_teacher_proposals[0].proposal_boxes])
                   dt_roih_logits = dynamic_teacher_model.roi_heads.box_predictor(dt_box_features.mean(dim=[2, 3]))[0]  # only class logits
                dynamic_teacher_pseudo_results = process_pseudo_label(dynamic_teacher_results, args.process_threshold)
                dynamic_teacher_pseudo_proposals_dict = instances_to_dict(dynamic_teacher_pseudo_results, logits=dt_roih_logits, type="All Proposals Dynamic")

                # Static teacher model inference (with weak augmentation)
                with torch.no_grad():
                     static_teacher_features, static_teacher_proposals, _, static_teacher_results, _ = static_teacher_model(data, mode="train", aug="weak", noise_std=args.noise_static_teacher, class_noise_prob=args.class_noise_prob_static_teacher, num_extra_boxes=args.num_extra_boxes_static_teacher)
                     st_box_features = static_teacher_model.roi_heads._shared_roi_transform([static_teacher_features['res4']], [static_teacher_proposals[0].proposal_boxes])
                     st_roih_logits = static_teacher_model.roi_heads.box_predictor(st_box_features.mean(dim=[2, 3]))[0]  # only class logits
                static_teacher_pseudo_results = process_pseudo_label(static_teacher_results, args.process_threshold)
                static_teacher_pseudo_proposals_dict = instances_to_dict(static_teacher_pseudo_results, logits=st_roih_logits, type="All Proposals Static")

                # Combine pseudo-labels from both teachers (consens mechanism)
                image_height, image_width = static_teacher_pseudo_results[0].image_size
                device = static_teacher_pseudo_results[0].gt_boxes.device
                my_selection_pseudo_labels_dict, selected_teacher_pseudo_results, dynamic_static_consens_mechanism = my_selection_pseudo_labels_process(
                    dynamic_teacher_pseudo_proposals_dict,
                    static_teacher_pseudo_proposals_dict,
                    args.configuration['categories_threshold'],
                    image_height,
                    image_width,
                    device,
                    args.similar_classes,
                    args.sigma,
                    args.similarity_threshold,
                    args.alpha,
                    args.beta,
                    args.boost_factor,
                    args.no_match_found,
                    args.non_maximum_threshold,
                    args.matching_weight,
                    args.num_classes,
                    args.redist
                )

                # Forward pass: Student model with pseudo-labels as targets
                s_features, s_proposals, proposal_losses, detector_losses = student_model(data, selected_teacher_pseudo_results, mode="train", aug="strong")


                with torch.no_grad():
                    s_box_features = student_model.roi_heads._shared_roi_transform([s_features['res4']], [s_proposals[0].proposal_boxes])
                    s_roih_logits = student_model.roi_heads.box_predictor(s_box_features.mean(dim=[2, 3]))
                student_class_logits = s_roih_logits[0]


                # Compute soft-label loss and entropy loss
                soft_label_kl_distillation_loss_value = soft_label_kl_distillation_loss(student_class_logits, selected_teacher_pseudo_results)
                stud_entropy_loss = entropy_loss(student_class_logits)
                #
                # # Feature Distillation Loss
                feat_dist_loss = feature_distillation_loss(dynamic_teacher_features, s_features)

                #Contrastive Loss
                contrastive_loss = compute_contrastive_loss(dynamic_teacher_features, s_features, temperature=args.contrastive_loss_temperature)


                if iteration%2 == 0:
                    clean_and_get_memory(logger=logger, clean=True, log=False)

                min_val = 1e-6
                max_val = 1e6
                losses = sum([
                    args.weight_proposal_loss * torch.clamp(sum(proposal_losses.values()), min=min_val, max=max_val),
                    args.weight_detector_loss * torch.clamp(sum(detector_losses.values()), min=min_val, max=max_val),
                    args.weight_soft_label_kl_distillation_loss_value * torch.clamp(soft_label_kl_distillation_loss_value, min=min_val, max=max_val),
                    args.weight_stud_entropy_loss * torch.clamp(stud_entropy_loss, min=min_val, max=max_val),
                    args.weight_feat_dist_loss * torch.clamp(feat_dist_loss, min=min_val, max=max_val),
                    args.weight_contrastive_loss * torch.clamp(contrastive_loss, min=min_val, max=max_val),
                ])

                # Dynamic scaling based on entropy and number of proposals
                number_proposals = len(my_selection_pseudo_labels_dict['bboxes'])
                proposal_factor = min(max(1, number_proposals), 50)
                probs = torch.softmax(dt_roih_logits, dim=-1)
                log_probs = torch.log_softmax(dt_roih_logits, dim=-1)
                entropy_per_proposal = - (probs * log_probs).sum(dim=-1)
                entropy_mean = entropy_per_proposal.mean()
                if args.factor_scaler:
                    factor_proposal = 1 + args.factor_scaler * proposal_factor * 2
                    factor_entropy = 1 + args.factor_scaler * entropy_mean.item()
                    losses = losses * factor_proposal * factor_entropy


                # Check loss validity and backpropagate and log learning rate and losses
                assert torch.isfinite(losses).all()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                if (iteration + 1) % args.batch_size == 0:
                   optimizer.step()
                   optimizer.zero_grad()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                periodic_checkpointer.step(iteration)

                logger.info(
                    f"epoch {epoch}, iteration {iteration} - {data[0]['image_id']} - Loss {losses}, "f"Factor Proposal {factor_proposal}, Factor Entropy {factor_entropy}: "
                    f"{', '.join([f'{name}: {value.item()}' for name, value in proposal_losses.items()])}, "
                    f"{', '.join([f'{name}: {value.item()}' for name, value in detector_losses.items()])} "
                    f"soft_label_loss: {soft_label_kl_distillation_loss_value.item()} "
                    f"stud_entropy_loss: {stud_entropy_loss.item()} "
                    f"feat_dist_loss: {feat_dist_loss.item()} "
                    f"contrastive_loss: {contrastive_loss.item()} "
                )

                if iteration%3 == 0:
                    clean_and_get_memory(logger=logger, clean=True, log=False)

                # Update dynamic Teacher (Exponential Model Average EMA)
                if iteration%args.update_periods == 0:
                    clean_and_get_memory(logger=logger, clean=True, log=True)
                    new_dyn_teacher_dict = update_teacher_model(student_model, dynamic_teacher_model, keep_rate=args.keep_rate_dyn)
                    dynamic_teacher_model.load_state_dict(new_dyn_teacher_dict)
                    clean_and_get_memory(logger=logger, clean=True, log=True)

            new_stat_teacher_dict = update_teacher_model(student_model, static_teacher_model, keep_rate=args.keep_rate_stat)
            static_teacher_model.load_state_dict(new_stat_teacher_dict)

            # End-of-epoch model evaluation
            logger.info(f"\nStudent model testing at end of Epoch {epoch}")
            student_model.eval()
            results = test_sfda(cfg, student_model)
            logger.info(results)

            logger.info(f"\nDynamic Teacher model testing at end of Epoch {epoch}")
            dynamic_teacher_model.eval()
            results = test_sfda(cfg, dynamic_teacher_model)
            logger.info(results)

            # End-of-epoch checkpoint saving
            torch.save(static_teacher_model.state_dict(), cfg.OUTPUT_DIR + "/static_teacher_model_{}.pth".format(epoch))
            torch.save(dynamic_teacher_model.state_dict(), cfg.OUTPUT_DIR + "/dynamic_teacher_model_{}.pth".format(epoch))
            torch.save(student_model.state_dict(), cfg.OUTPUT_DIR + "/student_model_{}.pth".format(epoch))

            clean_and_get_memory(logger=logger, clean=True, log=True)



def main(args: argparse.Namespace) -> None:
    """
    Description:
        Main function to initialize and train the SFDA model.
    Params:
        args: Parsed command-line arguments containing configuration and settings
    """
    cfg = load_detectron_config(args)
    args.logger.info(f"\n\ncfg: {cfg}\n")

    student_model, static_teacher_model, dynamic_teacher_model = load_models(args, cfg)
    student_model = student_model.to('cuda:1')

    # Load student and teacher models
    args.logger.info("Load Student, Dynamic Teacher and Static Teacher Model sucessfully")
    # Start training
    train_sfda(args, cfg, student_model, static_teacher_model, dynamic_teacher_model)


def load_detectron_config(args: argparse.Namespace) -> Any:
    """
    Description:
        load and configure the detectron2 configuration based on arguments
    Args:
        args: parsed command-line arguments containing configuration and settings
    Returns:
        cfg: configured detectron2 configuration object.
    """
    cfg = get_cfg()
    cfg.SOLVER.AMP.ENABLED = True
    cfg.SOLVER.BACKBONE_LR = args.backbone_lr
    cfg.SOLVER.ROI_HEAD_LR = args.roi_head_lr
    cfg.merge_from_file(args.configuration['config_file'])
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


if __name__ == "__main__":
    """Entry point for the script. Parses arguments and launches the training"""
    name = "dln_sci2fin"
    #name = "pln4_dln4"
    #name = "dln10_m6doc10"
    logger = get_logger(f"/workspace/log_{name}.txt")

    os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"

    configuration = SFDA_DLA_Configuration_Loader(name).get_configuration_dict()

    args = argparse.Namespace(
        dist_url='tcp://127.0.0.1:54160',
        eval_only=False,
        machine_rank=0,
        num_gpus=1,
        num_machines=1,
        opts=[],
        resume=False,
        logger=logger,
        configuration=configuration,
        total_epochs=10,
        seed=0,
        use_entropy_samling=False,
        test_periods=5000,
        update_periods=2000,
        process_threshold=0.01,
        contrastive_loss_temperature=0.07,
        factor_scaler=0.08, # or None
        batch_size=4,
        keep_rate_dyn=0.9999,
        keep_rate_stat=0.95,
        weight_proposal_loss=4,
        weight_detector_loss=4,
        weight_soft_label_kl_distillation_loss_value=3,
        weight_stud_entropy_loss=0.8,
        weight_feat_dist_loss=0.4,
        weight_contrastive_loss=0.1,
        score_for_selected_bbox=0.5,
        max_iou_value_threshold=0.9,
        #similar_classes=[0, 1, 3, 4, 5, 7, 9, 10], #0: Caption, 1: Footnote, 3: List - item, 4: Page - footer, 5: Page - header, 7: Section - header, 9: Text, 10: Title
        similar_classes=[],  #
        sigma=0.1,
        noise_static_teacher=2.0,
        noise_dynamic_teacher=3.0,
        class_noise_prob_static_teacher=0.01,
        class_noise_prob_dynamic_teacher=0.02,
        num_extra_boxes_static_teacher=1,
        num_extra_boxes_dynamic_teacher=1,
        similarity_threshold=0.7,
        alpha=0.6,
        beta=0.4,
        boost_factor=1.1,
        no_match_found=0.8,
        non_maximum_threshold=0.8,
        matching_weight=0.5,
        num_classes=11,
        redist=0.02,
        backbone_lr=0.0001,
        roi_head_lr=0.0001
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
