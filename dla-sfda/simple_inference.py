"""
File: dla-sfda/simple_inference.py
Description: Inference with given paths without overhead
"""

import json
from venv import logger
from tools.inference_helper import run_inference_with_evaluation




################ Evaluation DocLayNet and M6Doc with 10 Classes ###############

results = run_inference_with_evaluation(
    dataset_name="doclaynet_10_val",
    annotation_json_path="/extern_home/cvhci/temp/stewes/converted_coco_annotation_files/DLN_with_10_Classes_val.json",
    image_root_dln_path="/extern_home/cvhci/data/document_analysis/DocLayNet/PNG/",
    repo_source_root="/extern_home/home/stewes/source-free-domain-adaptive-document-layout-analysis/dla_sfda_docker_mapping/",
    model_config_yaml="dla-sfda/configs/doclaynet_10_sourcemodel.yaml",
    model_path="/extern_home/cvhci/temp/stewes/final_models/source_model_dln10.pth"
)
print(results)
logger.info(results)


results = run_inference_with_evaluation(
    dataset_name="m6doc_10_val",
    annotation_json_path="/extern_home/cvhci/temp/stewes/converted_coco_annotation_files/M6Doc_with_10_Classes_val.json",
    image_root_dln_path="/extern_home/cvhci/data/document_analysis/M6Doc/val2017/",
    repo_source_root="/extern_home/home/stewes/source-free-domain-adaptive-document-layout-analysis/dla_sfda_docker_mapping/",
    model_config_yaml="dla-sfda/configs/doclaynet_10_sourcemodel.yaml",
    model_path="/extern_home/cvhci/temp/stewes/final_models/source_model_dln10.pth"
)
print(results)
logger.info(results)

results = run_inference_with_evaluation(
    dataset_name="m6doc_10_val",
    annotation_json_path="/extern_home/cvhci/temp/stewes/converted_coco_annotation_files/M6Doc_with_10_Classes_val.json",
    image_root_dln_path="/extern_home/cvhci/data/document_analysis/M6Doc/val2017/",
    repo_source_root="/extern_home/home/stewes/source-free-domain-adaptive-document-layout-analysis/dla_sfda_docker_mapping/",
    model_config_yaml="dla-sfda/configs/m6doc_10_sourcemodel.yaml",
    model_path="/extern_home/cvhci/temp/stewes/final_models/source_model_m6doc10.pth"
)
print(results)
logger.info(results)

results = run_inference_with_evaluation(
    dataset_name="doclaynet_10_val",
    annotation_json_path="/extern_home/cvhci/temp/stewes/converted_coco_annotation_files/DLN_with_10_Classes_val.json",
    image_root_dln_path="/extern_home/cvhci/data/document_analysis/DocLayNet/PNG/",
    repo_source_root="/extern_home/home/stewes/source-free-domain-adaptive-document-layout-analysis/dla_sfda_docker_mapping/",
    model_config_yaml="dla-sfda/configs/m6doc_10_sourcemodel.yaml",
    model_path="/extern_home/cvhci/temp/stewes/final_models/source_model_m6doc10.pth"
)
print(results)
logger.info(results)



################ Evaluation PubLayNet and DocLayNet with 4 Classes ###############


results = run_inference_with_evaluation(
    dataset_name="publaynet4_val",
    annotation_json_path="/extern_home/cvhci/temp/stewes/converted_coco_annotation_files/PLN_with_4_Classes_val.json",
    image_root_dln_path="/extern_home/cvhci/data/Documents_Datasets/PubLayNet/publaynet/val/",
    repo_source_root="/extern_home/home/stewes/source-free-domain-adaptive-document-layout-analysis/dla_sfda_docker_mapping/",
    model_config_yaml="dla-sfda/configs/publaynet4_sourcemodel.yaml",
    model_path="/extern_home/cvhci/temp/stewes/final_models/source_model_pln4.pth"
)
print(results)
logger.info(results)

results = run_inference_with_evaluation(
    dataset_name="doclaynet4_val",
    annotation_json_path="/extern_home/cvhci/temp/stewes/converted_coco_annotation_files/DLN_with_4_Classes_val.json",
    image_root_dln_path="/extern_home/cvhci/data/document_analysis/DocLayNet/PNG/",
    repo_source_root="/extern_home/home/stewes/source-free-domain-adaptive-document-layout-analysis/dla_sfda_docker_mapping/",
    model_config_yaml="dla-sfda/configs/publaynet4_sourcemodel.yaml",
    model_path="/extern_home/cvhci/temp/stewes/final_models/source_model_pln4.pth"
)
print(results)
logger.info(results)

results = run_inference_with_evaluation(
    dataset_name="doclaynet4_val",
    annotation_json_path="/extern_home/cvhci/temp/stewes/converted_coco_annotation_files/DLN_with_4_Classes_val.json",
    image_root_dln_path="/extern_home/cvhci/data/document_analysis/DocLayNet/PNG/",
    repo_source_root="/extern_home/home/stewes/source-free-domain-adaptive-document-layout-analysis/dla_sfda_docker_mapping/",
    model_config_yaml="dla-sfda/configs/doclaynet_4_sourcemodel.yaml",
    model_path="/extern_home/cvhci/temp/stewes/final_models/source_model_dln4.pth"
)
print(results)
logger.info(results)


results = run_inference_with_evaluation(
    dataset_name="publaynet4_val",
    annotation_json_path="/extern_home/cvhci/temp/stewes/converted_coco_annotation_files/PLN_with_4_Classes_val.json",
    image_root_dln_path="/extern_home/cvhci/data/Documents_Datasets/PubLayNet/publaynet/val/",
    repo_source_root="/extern_home/home/stewes/source-free-domain-adaptive-document-layout-analysis/dla_sfda_docker_mapping/",
    model_config_yaml="dla-sfda/configs/doclaynet_4_sourcemodel.yaml",
    model_path="/extern_home/cvhci/temp/stewes/final_models/source_model_dln4.pth"
)
print(results)
logger.info(results)




################ Evaluation DocLayNet Subcategories with 11 Classes ###############


model_config_yaml = "dla-sfda/configs/rcnn_11_model.yaml"
repo_source_root = "/extern_home/home/stewes/source-free-domain-adaptive-document-layout-analysis/dla_sfda_docker_mapping/"
image_root_dln_path = "/extern_home/cvhci/data/document_analysis/DocLayNet/PNG/"



dataset_dict_list = [
    {
        "dataset_name": "doclaynet_sci_val",
        "image_root_path": image_root_dln_path,
        "annotation_json_path": "/extern_home/cvhci/temp/stewes/DocLayNetSubcats/COCO/scientific_articles_val.json"
    },
    {
        "dataset_name": "doclaynet_fin_val",
        "image_root_path": image_root_dln_path,
        "annotation_json_path": "/extern_home/cvhci/temp/stewes/DocLayNetSubcats/COCO/financial_reports_val.json"
    },
    {
        "dataset_name": "doclaynet_man_val",
        "image_root_path": image_root_dln_path,
        "annotation_json_path": "/extern_home/cvhci/temp/stewes/DocLayNetSubcats/COCO/manuals_val.json"
    },
    {
        "dataset_name": "doclaynet_law_val",
        "image_root_path": image_root_dln_path,
        "annotation_json_path": "/extern_home/cvhci/temp/stewes/DocLayNetSubcats/COCO/laws_and_regulations_val.json"
    }
]

model_dict_list = [
    {
        "model_name": "source_model_dln_sci",
        "model_path": "/extern_home/cvhci/temp/stewes/final_models/source_model_dln_sci.pth"
    },
    {
        "model_name": "source_model_dln_fin",
        "model_path": "/extern_home/cvhci/temp/stewes/final_models/source_model_dln_fin.pth"
    },
    {
        "model_name": "source_model_dln_man",
        "model_path": "/extern_home/cvhci/temp/stewes/final_models/source_model_dln_man.pth"
    },
    {
        "model_name": "source_model_dln_law",
        "model_path": "/extern_home/cvhci/temp/stewes/final_models/source_model_dln_law.pth"
    }
]

results_dict = {}
for model_element in model_dict_list:
    model_results_dict = {}
    for dataset_element in dataset_dict_list:
        results = run_inference_with_evaluation(
            dataset_name=dataset_element['dataset_name'],
            annotation_json_path=dataset_element['annotation_json_path'],
            image_root_dln_path=dataset_element['image_root_path'],
            repo_source_root=repo_source_root,
            model_config_yaml=model_config_yaml,
            model_path=model_element['model_path']
        )

        print(f"\t{model_element['model_name']} - {dataset_element['dataset_name']}: {results}")
        logger.info(f"\t{model_element['model_name']} - {dataset_element['dataset_name']}: {results}")

        model_results_dict[dataset_element['dataset_name']] = {
            'general_aps': {key: value for key, value in results['bbox'].items() if '-' not in key},
            'class_aps': {key: value for key, value in results['bbox'].items() if '-' in key}
        }
    results_dict[model_element['model_name']] = model_results_dict

print(results_dict)
logger.info(results_dict)

with open("/workspace/results_dict.json", "w") as json_file:
    json.dump(results_dict, json_file, indent=4)



################ Evaluation After SF-DLA ###############

results = run_inference_with_evaluation( # DLN-SCI->DLN-FIN
    dataset_name="doclaynet_fin_val",
    annotation_json_path="/extern_home/cvhci/temp/stewes/DocLayNetSubcats/COCO/financial_reports_val.json",
    image_root_dln_path="/extern_home/cvhci/data/document_analysis/DocLayNet/PNG/",
    repo_source_root="/extern_home/home/stewes/source-free-domain-adaptive-document-layout-analysis/dla_sfda_docker_mapping/",
    model_config_yaml="dla-sfda/configs/rcnn_11_model.yaml",
    model_path="/extern_home/cvhci/temp/stewes/final_models/result_model_dln_sci2fin.pth"
)
print(results)
logger.info(results)

results = run_inference_with_evaluation( # DLN-MAN->DLN-FIN
    dataset_name="doclaynet_fin_val",
    annotation_json_path="/extern_home/cvhci/temp/stewes/DocLayNetSubcats/COCO/financial_reports_val.json",
    image_root_dln_path="/extern_home/cvhci/data/document_analysis/DocLayNet/PNG/",
    repo_source_root="/extern_home/home/stewes/source-free-domain-adaptive-document-layout-analysis/dla_sfda_docker_mapping/",
    model_config_yaml="dla-sfda/configs/rcnn_11_model.yaml",
    model_path="/extern_home/cvhci/temp/stewes/final_models/result_model_dln_man2fin.pth"
)
print(results)
logger.info(results)

results = run_inference_with_evaluation( # DLN-LAW->DLN-MAN
    dataset_name="doclaynet_man_val",
    annotation_json_path="/extern_home/cvhci/temp/stewes/DocLayNetSubcats/COCO/manuals_val.json",
    image_root_dln_path="/extern_home/cvhci/data/document_analysis/DocLayNet/PNG/",
    repo_source_root="/extern_home/home/stewes/source-free-domain-adaptive-document-layout-analysis/dla_sfda_docker_mapping/",
    model_config_yaml="dla-sfda/configs/rcnn_11_model.yaml",
    model_path="/extern_home/cvhci/temp/stewes/final_models/result_model_dln_law2man.pth"
)
print(results)
logger.info(results)



results = run_inference_with_evaluation( # PLN-4 -> DLN-4
    dataset_name="doclaynet4_val",
    annotation_json_path="/extern_home/cvhci/temp/stewes/converted_coco_annotation_files/DLN_with_4_Classes_val.json",
    image_root_dln_path="/extern_home/cvhci/data/document_analysis/DocLayNet/PNG/",
    repo_source_root="/extern_home/home/stewes/source-free-domain-adaptive-document-layout-analysis/dla_sfda_docker_mapping/",
    model_config_yaml="dla-sfda/configs/publaynet4_sourcemodel.yaml",
    model_path="/extern_home/cvhci/temp/stewes/final_models/result_model_pln4_dln4.pth"
)
print(results)
logger.info(results)


results = run_inference_with_evaluation( # DLN-10 -> M6Doc-10
    dataset_name="m6doc_10_val",
    annotation_json_path="/extern_home/cvhci/temp/stewes/converted_coco_annotation_files/M6Doc_with_10_Classes_val.json",
    image_root_dln_path="/extern_home/cvhci/data/document_analysis/M6Doc/val2017/",
    repo_source_root="/extern_home/home/stewes/source-free-domain-adaptive-document-layout-analysis/dla_sfda_docker_mapping/",
    model_config_yaml="dla-sfda/configs/doclaynet_10_sourcemodel.yaml",
    model_path="/extern_home/cvhci/temp/stewes/final_models/result_model_dln10_m6doc.pth"
)
print(results)
logger.info(results)