"""
File: dla-sfda/configs/SourceModelConfig.py
Description: The configurations for training the source models are located here
"""

class SourceModelConfig:
    """
    Description:
        This class collects the paths to the data, models and configurations for the configuration name fr the Source Model Training
    Params:
        name (str): name of the configuration
    """
    def __init__(
        self,
        name: str
    ):
        known_config_names = ["dln_sci", "dln_fin", "dln_man", "dln_law", "dln4", "pln4", "dln10", "m6doc10"]
        if name in known_config_names:
            self.name = name
        else:
            raise Exception(f"configname {name} not in known configs ({known_config_names})")

        self.__path_to_dla_sfda_repo = "/extern_home/home/stewes/source-free-domain-adaptive-document-layout-analysis/"
        self.__path_to_model_dir = "/extern_home/cvhci/temp/stewes/path_folder/"
        self.__path_to_datasets = "/extern_home/cvhci/"
        self.__path_dln_image_root = f'{self.__path_to_datasets}data/document_analysis/DocLayNet/PNG/'

    def __get_dln_image_root_and_names(self):
        return self.__path_dln_image_root, self.__path_dln_image_root, "doclaynet_train", "doclaynet_val"

    def __get_dln4_paths(self):
        train_dataset_annotation_path = f'{self.__path_to_datasets}temp/stewes/converted_coco_annotation_files/DLN_with_4_Classes_train.json'
        val_dataset_annotation_path = f'{self.__path_to_datasets}temp/stewes/converted_coco_annotation_files/DLN_with_4_Classes_val.json'
        return train_dataset_annotation_path, val_dataset_annotation_path

    def __get_dln10_paths(self):
        train_dataset_annotation_path = f'{self.__path_to_datasets}temp/stewes/converted_coco_annotation_files/DLN_with_10_Classes_train.json'
        val_dataset_annotation_path = f'{self.__path_to_datasets}temp/stewes/converted_coco_annotation_files/DLN_with_10_Classes_val.json'
        return train_dataset_annotation_path, val_dataset_annotation_path

    def __get_dln_subcategory(self, target_dataset_name: str):
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

    def __get_dataset_pln4(self):
        train_dataset_name = "publaynet4_train"
        test_dataset_name = "publaynet4_test"
        train_dataset_annotation_path = f'{self.__path_to_datasets}temp/stewes/converted_coco_annotation_files/PLN_with_4_Classes_train.json'
        test_dataset_annotation_path = f'{self.__path_to_datasets}temp/stewes/converted_coco_annotation_files/PLN_with_4_Classes_val.json'
        train_dataset_image_folder_path = f'{self.__path_to_datasets}data/Documents_Datasets/PubLayNet/publaynet/train/'
        test_dataset_image_folder_path = f'{self.__path_to_datasets}data/Documents_Datasets/PubLayNet/publaynet/val/'
        return train_dataset_name, test_dataset_name, train_dataset_annotation_path, test_dataset_annotation_path, train_dataset_image_folder_path, test_dataset_image_folder_path

    def __get_dataset_m6doc10(self):
        train_dataset_name = "m6doc_10_train"
        test_dataset_name = "m6doc_10_val"
        train_dataset_annotation_path = f'{self.__path_to_datasets}temp/stewes/converted_coco_annotation_files/M6Doc_with_10_Classes_train.json'
        test_dataset_annotation_path = f'{self.__path_to_datasets}temp/stewes/converted_coco_annotation_files/M6Doc_with_10_Classes_val.json'
        train_dataset_image_folder_path = f'{self.__path_to_datasets}data/document_analysis/M6Doc/train2017/'
        test_dataset_image_folder_path = f'{self.__path_to_datasets}data/document_analysis/M6Doc/val2017/'
        return train_dataset_name, test_dataset_name, train_dataset_annotation_path, test_dataset_annotation_path, train_dataset_image_folder_path, test_dataset_image_folder_path



    def get_configuration_dict(self):
        """
        Description:
            This method collects the paths to the data, models and configurations for the configuration of this class
        Return:
            dict with config_file, train_dataset_name, val_dataset_name, train_dataset_annotation_path, val_dataset_annotation_path, train_dataset_image_folder_path, val_dataset_image_folder_path
        """
        if self.name == "dln4":
            config_file = f'{self.__path_to_dla_sfda_repo}dla_sfda_docker_mapping/dla-sfda/configs/doclaynet_4_sourcemodel.yaml'
            train_dataset_image_folder_path, val_dataset_image_folder_path, train_dataset_name, val_dataset_name = self.__get_dln_image_root_and_names()
            train_dataset_annotation_path, val_dataset_annotation_path = self.__get_dln4_paths()
        elif self.name == "pln4":
            config_file = f'{self.__path_to_dla_sfda_repo}dla_sfda_docker_mapping/dla-sfda/configs/publaynet4_sourcemodel.yaml'
            train_dataset_name, val_dataset_name, train_dataset_annotation_path, val_dataset_annotation_path, train_dataset_image_folder_path, val_dataset_image_folder_path = self.__get_dataset_pln4()
        elif self.name == "dln10":
            config_file = f'{self.__path_to_dla_sfda_repo}dla_sfda_docker_mapping/dla-sfda/configs/doclaynet_10_sourcemodel.yaml'
            train_dataset_image_folder_path, val_dataset_image_folder_path, train_dataset_name, val_dataset_name = self.__get_dln_image_root_and_names()
            train_dataset_annotation_path, val_dataset_annotation_path = self.__get_dln10_paths()
        elif self.name == "m6doc10":
            config_file = f'{self.__path_to_dla_sfda_repo}dla_sfda_docker_mapping/dla-sfda/configs/m6doc_10_sourcemodel.yaml'
            train_dataset_name, val_dataset_name, train_dataset_annotation_path, val_dataset_annotation_path, train_dataset_image_folder_path, val_dataset_image_folder_path = self.__get_dataset_m6doc10()
        elif "dln" in self.name and len(self.name.split("_")) > 1 :
            config_file = f'{self.__path_to_dla_sfda_repo}dla_sfda_docker_mapping/dla-sfda/configs/doclaynet_11_sourcemodel.yaml'
            train_dataset_image_folder_path, val_dataset_image_folder_path, train_dataset_name, val_dataset_name = self.__get_dln_image_root_and_names()
            target_dataset = self.name.split("_")[-1]
            if target_dataset in ["sci", "fin", "man", "law"]:
                train_dataset_annotation_path, val_dataset_annotation_path = self.__get_dln_subcategory(target_dataset_name=f"dln_{target_dataset}")
            else:
                raise Exception(f"target dataset {target_dataset} not valid")
        else:
            raise Exception(f"name {self.name} not valid")

        return {
            "name": self.name,
            "config_file": config_file,
            "train_dataset_name": train_dataset_name,
            "val_dataset_name": val_dataset_name,
            "train_dataset_annotation_path": train_dataset_annotation_path,
            "val_dataset_annotation_path": val_dataset_annotation_path,
            "train_dataset_image_folder_path": train_dataset_image_folder_path,
            "val_dataset_image_folder_path": val_dataset_image_folder_path,
        }
