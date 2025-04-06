"""
File: dla-sfda/configs/SFDA_DLA_Configuration_Loader.py
Description: The configurations for the SFDA are located here
"""
from typing import Dict
from detectron2.data.datasets import register_coco_instances

dln_categories_threshold = {
    0: {"S": 0.98, "M": 0.98, "L": 0.98, "XL": 0.98},  # 'Caption'
    1: {"S": 0.98, "M": 0.98, "L": 0.98, "XL": 0.98},  # 'Footnote'
    2: {"S": 0.98, "M": 0.98, "L": 0.98, "XL": 0.98},  # 'Formula'
    3: {"S": 0.98, "M": 0.98, "L": 0.8, "XL": 0.8},  # 'List-item'
    4: {"S": 0.3, "M": 0.98, "L": 0.98, "XL": 0.98},  # 'Page-footer
    5: {"S": 0.8, "M": 0.8, "L": 0.8, "XL": 0.98},  # 'Page-header'
    6: {"S": 0.98, "M": 0.98, "L": 0.98, "XL": 0.9},  # 'Picture'
    7: {"S": 0.4, "M": 0.6, "L": 0.8, "XL": 0.98},  # 'Section-header'
    8: {"S": 0.98, "M": 0.98, "L": 0.98, "XL": 0.7},  # 'Table'
    9: {"S": 0.98, "M": 0.98, "L": 0.8, "XL": 0.6},  # 'Text'
    10: {"S": 0.98, "M": 0.98, "L": 0.98, "XL": 0.98}  # 'Title'
}
dln_class_id_name_mapping = {
    0: 'Caption',
    1: 'Footnote',
    2: 'Formula',
    3: 'List-item',
    4: 'Page-footer',
    5: 'Page-header',
    6: 'Picture',
    7: 'Section-header',
    8: 'Table',
    9: 'Text',
    10: 'Title'
}

pln4dln4_categories_threshold = {
    0: {"S": 0.8, "M": 0.8, "L": 0.7, "XL": 0.9},  # 'text'
    1: {"S": 0.7, "M": 0.5, "L": 0.5, "XL": 0.5},  # 'title'
    2: {"S": 0.5, "M": 0.5, "L": 0.5, "XL": 0.9},  # 'table'
    3: {"S": 0.5, "M": 0.5, "L": 0.5, "XL": 0.9},  # 'figure'
}
pln4dln4_class_id_name_mapping = {
    0: 'text',
    1: 'title',
    2: 'table',
    3: 'figure'
}

dln10m6doc10_categories_threshold = {
    0: {"S": 0.4, "M": 0.4, "L": 0.4, "XL": 0.4},  # 'Caption'
    1: {"S": 0.2, "M": 0.2, "L": 0.8, "XL": 0.8},  # 'Footnote'
    2: {"S": 0.9, "M": 0.9, "L": 0.9, "XL": 0.9},  # 'Formula'
    3: {"S": 0.8, "M": 0.5, "L": 0.5, "XL": 0.5},  # 'Page-footer'
    4: {"S": 0.6, "M": 0.9, "L": 0.9, "XL": 0.9},  # 'Page-header'
    5: {"S": 0.8, "M": 0.8, "L": 0.6, "XL": 0.8},  # 'Picture'
    6: {"S": 0.6, "M": 0.5, "L": 0.5, "XL": 0.5},  # 'Section-header'
    7: {"S": 0.8, "M": 0.8, "L": 0.5, "XL": 0.8},  # 'Table'
    8: {"S": 0.8, "M": 0.7, "L": 0.7, "XL": 0.7},  # 'Text'
    9: {"S": 0.4, "M": 0.4, "L": 0.4, "XL": 0.4},  # 'Title'
}
dln10m6doc10_class_id_name_mapping = {
    0: 'Caption',
    1: 'Footnote',
    2: 'Formula',
    3: 'Page-footer',
    4: 'Page-header',
    5: 'Picture',
    6: 'Section-header',
    7: 'Table',
    8: 'Text',
    9: 'Title'
}




def get_size_category(area: int) -> str:
    """
    Description:
        Mapping of the bounding box size to size categories
    Params:
        area (int): bounding box area value
    Return:
        size category of bounding box (S, M, L, XL)
    """
    if area < 3400:
        size = "S"
    elif area < 6500:
        size = "M"
    elif area < 13200:
        size = "L"
    else:
        size = "XL"
    return size


class SFDA_DLA_Configuration_Loader:
    """
    Description:
        This class collects the paths to the data, models and configurations for the configuration name for the SFDA Training
    Params:
        name (str): name of the configuration
    """

    def __init__(
            self,
            name: str
    ):
        self.name = name
        self.__path_to_dla_sfda_repo = "/extern_home/home/stewes/source-free-domain-adaptive-document-layout-analysis/"
        self.__path_to_model_dir = "/extern_home/cvhci/temp/stewes/final_models/"
        self.__path_to_datasets = "/extern_home/cvhci/"
        self.__path_dln_image_root = f'{self.__path_to_datasets}data/document_analysis/DocLayNet/PNG/'

    def __get_dln_image_root_and_names(self):
        return self.__path_dln_image_root, self.__path_dln_image_root, "doclaynet_train", "doclaynet_val"

    def __get_dln_subcategory(self, target_dataset_name):
        if target_dataset_name == "dln_sci":
            train_dataset_annotation_path = f'{self.__path_to_datasets}temp/stewes/DocLayNetSubcats/COCO/scientific_articles_train.json'
            val_dataset_annotation_path = f'{self.__path_to_datasets}temp/stewes/DocLayNetSubcats/COCO/scientific_articles_val.json'
            return train_dataset_annotation_path, val_dataset_annotation_path
        elif target_dataset_name == "dln_fin":
            train_dataset_annotation_path = f'{self.__path_to_datasets}temp/stewes/DocLayNetSubcats/COCO/financial_reports_train.json'
            val_dataset_annotation_path = f'{self.__path_to_datasets}temp/stewes/DocLayNetSubcats/COCO/financial_reports_val.json'
            return train_dataset_annotation_path, val_dataset_annotation_path
        elif target_dataset_name == "dln_man":
            train_dataset_annotation_path = f'{self.__path_to_datasets}temp/stewes/DocLayNetSubcats/COCO/manuals_train.json'
            val_dataset_annotation_path = f'{self.__path_to_datasets}temp/stewes/DocLayNetSubcats/COCO/manuals_val.json'
            return train_dataset_annotation_path, val_dataset_annotation_path
        elif target_dataset_name == "dln_law":
            train_dataset_annotation_path = f'{self.__path_to_datasets}temp/stewes/DocLayNetSubcats/COCO/laws_and_regulations_train.json'
            val_dataset_annotation_path = f'{self.__path_to_datasets}temp/stewes/DocLayNetSubcats/COCO/laws_and_regulations_val.json'
            return train_dataset_annotation_path, val_dataset_annotation_path
        else:
            raise Exception(f"Error: {target_dataset_name} not available")

    def __get_source_model_paths(self, source_model_name):
        if source_model_name == "dln_sci":
            return f'{self.__path_to_model_dir}source_model_dln_sci.pth'
        elif source_model_name == "dln_fin":
            return f'{self.__path_to_model_dir}source_model_dln_fin.pth'
        elif source_model_name == "dln_man":
            return f'{self.__path_to_model_dir}source_model_dln_man.pth'
        elif source_model_name == "dln_law":
            return f'{self.__path_to_model_dir}source_model_dln_law.pth'
        else:
            raise Exception(f"Error: {source_model_name} not available")

    def get_configuration_dict(self) -> Dict:
        """
        Description:
            This method collects the paths to the data, models and configurations for the configuration of this class
        Return:
            dict with name, config_file, model_dir, train_dataset_name, val_dataset_name, train_dataset_annotation_path, val_dataset_annotation_path, train_dataset_image_folder_path, val_dataset_image_folder_path, categories_threshold, class_id_name_mapping
        """
        if self.name == "pln4_dln4":
            config_file = f'{self.__path_to_dla_sfda_repo}dla_sfda_docker_mapping/dla-sfda/configs/sfda_pln4dln4.yaml'
            model_dir = "/extern_home/cvhci/temp/stewes/final_models/source_model_pln4.pth"
            train_dataset_name = "doclaynet4_train"
            val_dataset_name = "doclaynet4_val"
            train_dataset_annotation_path = "/extern_home/cvhci/temp/stewes/converted_coco_annotation_files/DLN_with_4_Classes_train.json"
            train_dataset_image_folder_path = "/extern_home/cvhci/data/document_analysis/DocLayNet/PNG/"
            val_dataset_annotation_path = "/extern_home/cvhci/temp/stewes/converted_coco_annotation_files/DLN_with_4_Classes_val.json"
            val_dataset_image_folder_path = "/extern_home/cvhci/data/document_analysis/DocLayNet/PNG/"

            register_coco_instances(train_dataset_name, {}, train_dataset_annotation_path, train_dataset_image_folder_path)
            register_coco_instances(val_dataset_name, {}, val_dataset_annotation_path, val_dataset_image_folder_path)

            return {
                "name": self.name,
                "config_file": config_file,
                "model_dir": model_dir,
                "train_dataset_name": train_dataset_name,
                "val_dataset_name": val_dataset_name,
                "train_dataset_annotation_path": train_dataset_annotation_path,
                "val_dataset_annotation_path": val_dataset_annotation_path,
                "train_dataset_image_folder_path": train_dataset_image_folder_path,
                "val_dataset_image_folder_path": val_dataset_image_folder_path,
                "categories_threshold": pln4dln4_categories_threshold,
                "class_id_name_mapping": pln4dln4_class_id_name_mapping
            }
        elif self.name == "dln10_m6doc10":
            config_file = f'{self.__path_to_dla_sfda_repo}dla_sfda_docker_mapping/dla-sfda/configs/sfda_dln10m6doc10.yaml'
            #model_dir = "/extern_home/cvhci/temp/stewes/path_folder/source_models/source_model_dln10_new_setting.pth"
            model_dir = "/extern_home/cvhci/temp/stewes/final_models/source_model_dln10.pth"
            train_dataset_name = "m6doc10_train"
            val_dataset_name = "m6doc10_val"
            train_dataset_annotation_path = "/extern_home/cvhci/temp/stewes/converted_coco_annotation_files/M6Doc_with_10_Classes_train.json"
            train_dataset_image_folder_path = "/extern_home/cvhci/data/document_analysis/M6Doc/train2017/"
            val_dataset_annotation_path = "/extern_home/cvhci/temp/stewes/converted_coco_annotation_files/M6Doc_with_10_Classes_val.json"
            val_dataset_image_folder_path = "/extern_home/cvhci/data/document_analysis/M6Doc/val2017/"

            register_coco_instances(train_dataset_name, {}, train_dataset_annotation_path, train_dataset_image_folder_path)
            register_coco_instances(val_dataset_name, {}, val_dataset_annotation_path, val_dataset_image_folder_path)

            return {
                "name": self.name,
                "config_file": config_file,
                "model_dir": model_dir,
                "train_dataset_name": train_dataset_name,
                "val_dataset_name": val_dataset_name,
                "train_dataset_annotation_path": train_dataset_annotation_path,
                "val_dataset_annotation_path": val_dataset_annotation_path,
                "train_dataset_image_folder_path": train_dataset_image_folder_path,
                "val_dataset_image_folder_path": val_dataset_image_folder_path,
                "categories_threshold": dln10m6doc10_categories_threshold,
                "class_id_name_mapping": dln10m6doc10_class_id_name_mapping
            }
        elif "dln" in self.name:
            config_file = f'{self.__path_to_dla_sfda_repo}dla_sfda_docker_mapping/dla-sfda/configs/sfda_dln.yaml'
            train_dataset_image_folder_path, val_dataset_image_folder_path, train_dataset_name, val_dataset_name = self.__get_dln_image_root_and_names()
            source_model = self.name.split("_")[-1].split("2")[0]
            target_dataset = self.name.split("_")[-1].split("2")[1]

            if source_model not in ["sci", "fin", "man", "law"]:
                raise Exception(f"source model {source_model} not valid")

            if target_dataset not in ["sci", "fin", "man", "law"]:
                raise Exception(f"target dataset {target_dataset} not valid")

            model_dir = self.__get_source_model_paths(source_model_name=f"dln_{source_model}")
            train_dataset_annotation_path, val_dataset_annotation_path, = self.__get_dln_subcategory(target_dataset_name=f"dln_{target_dataset}")

            register_coco_instances(train_dataset_name, {}, train_dataset_annotation_path, train_dataset_image_folder_path)
            register_coco_instances(val_dataset_name, {}, val_dataset_annotation_path, val_dataset_image_folder_path)

            return {
                "name": self.name,
                "config_file": config_file,
                "model_dir": model_dir,
                "train_dataset_name": train_dataset_name,
                "val_dataset_name": val_dataset_name,
                "train_dataset_annotation_path": train_dataset_annotation_path,
                "val_dataset_annotation_path": val_dataset_annotation_path,
                "train_dataset_image_folder_path": train_dataset_image_folder_path,
                "val_dataset_image_folder_path": val_dataset_image_folder_path,
                "categories_threshold": dln_categories_threshold,
                "class_id_name_mapping": dln_class_id_name_mapping
            }
        else:
            raise Exception(f"dln not in {self.name}")
