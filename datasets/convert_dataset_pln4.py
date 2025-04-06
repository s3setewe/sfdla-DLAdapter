"""
File: dla-sfda/datasets/convert_dataset_pln4.py
Description: Create Dataset PLN-4
"""

import json
from typing import List


def convert_pln5_to_pln4(source_basic_path: str, target_basic_path: str, split: str, new_categories_pln4_dln4: List[dict]) -> None:
    """
    This function converts the PubLayNet Dataset (5 Classes) to the version with 4 classes (PLN-4)
    Args:
        source_basic_path: source path of your PLN Annotation Json
        target_basic_path: target path of the converted Annotation Json
        split: train, val, test
        new_categories_pln4_dln4: category list_dict of MS COCO
    """

    file_path = f'{source_basic_path}publaynet/{split}.json'
    with open(file_path, 'r') as file:
        original_annotation_COCO = json.load(file)
    print(f"Read {file_path}")


    original_annotation_COCO["categories"] = new_categories_pln4_dln4

    for ann_element in original_annotation_COCO['annotations']:
        ann_element.pop('segmentation')
        if ann_element['category_id'] == 1:
            pass
        elif ann_element['category_id'] == 2:
            pass
        elif ann_element['category_id'] == 3:
            ann_element['category_id'] = 1
        elif ann_element['category_id'] == 4:
            ann_element['category_id'] = 3
        elif ann_element['category_id'] == 5:
            ann_element['category_id'] = 4
        else:
            print("Error")

    for values_dict in original_annotation_COCO['images']:
        keys_to_remove = [key for key in values_dict.keys() if key not in ['file_name', 'width', 'id', 'height']]
        for key in keys_to_remove:
            values_dict.pop(key)

    save_path = f"{target_basic_path}PLN_with_4_Classes_{split}.json"
    with open(save_path, "w") as outfile:
        json.dump(original_annotation_COCO, outfile)
    print(f"Saved {save_path}")



old_categories_pln5 = [ # not needed, only for comparison
         {'supercategory': '', 'id': 1, 'name': 'text'},
         {'supercategory': '', 'id': 2, 'name': 'title'},
         {'supercategory': '', 'id': 3, 'name': 'list'},
         {'supercategory': '', 'id': 4, 'name': 'table'},
         {'supercategory': '', 'id': 5, 'name': 'figure'}
    ]

new_categories_pln4_dln4 = [
        {'supercategory': '', 'id': 1, 'name': 'text'},
        {'supercategory': '', 'id': 2, 'name': 'title'},
        {'supercategory': '', 'id': 3, 'name': 'table'},
        {'supercategory': '', 'id': 4, 'name': 'figure'}
    ]

source_basic_path = "/extern_home/cvhci/data/Documents_Datasets/PubLayNet/"
target_basic_path = "/"

convert_pln5_to_pln4(source_basic_path=source_basic_path, target_basic_path=target_basic_path, split="val", new_categories_pln4_dln4=new_categories_pln4_dln4)
convert_pln5_to_pln4(source_basic_path=source_basic_path, target_basic_path=target_basic_path, split="test", new_categories_pln4_dln4=new_categories_pln4_dln4)
convert_pln5_to_pln4(source_basic_path=source_basic_path, target_basic_path=target_basic_path, split="train", new_categories_pln4_dln4=new_categories_pln4_dln4)
