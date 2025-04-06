"""
File: dla-sfda/datasets/convert_dataset_dln4.py
Description: Create Dataset DLN-4
"""

import json
from typing import List


def convert_dln11_to_dln4(source_basic_path: str, target_basic_path: str, split: str, new_categories_pln4_dln4: List[dict], category_mapping_pln4_dln4: dict) -> None:
    """
    This function converts the DocLayNet Dataset (11 Classes) to the version with 4 classes (DLN-4)
    Args:
        source_basic_path: source path of your DLN Annotation Json
        target_basic_path: target path of the converted Annotation Json
        split: train, val, test
        new_categories_pln4_dln4: category list_dict of MS COCO
        category_mapping_pln4_dln4: mapping dict
    """
    file_path = f'{source_basic_path}{split}.json'
    with open(file_path, 'r') as file:
        original_annotation_COCO = json.load(file)
    print(f"Read {file_path}")

    original_annotation_COCO["categories"] = new_categories_pln4_dln4
    filtered_annotations = []
    for ann_element in original_annotation_COCO['annotations']:
        original_id = ann_element['category_id']
        if original_id in category_mapping_pln4_dln4:
            ann_element['category_id'] = category_mapping_pln4_dln4[original_id]
            filtered_annotations.append(ann_element)
        elif original_id not in [5, 6]:
            print("Error:", original_id)
    original_annotation_COCO['annotations'] = filtered_annotations

    save_path = f"{target_basic_path}DLN_with_4_Classes_{split}.json"
    with open(save_path, "w") as outfile:
        json.dump(original_annotation_COCO, outfile)
    print(f"Saved {save_path}")


old_categories_dln11 = [ # not needed, only for comparison
    {'supercategory': 'Caption', 'id': 1, 'name': 'Caption'},
    {'supercategory': 'Footnote', 'id': 2, 'name': 'Footnote'},
    {'supercategory': 'Formula', 'id': 3, 'name': 'Formula'},
    {'supercategory': 'List-item', 'id': 4, 'name': 'List-item'},
    {'supercategory': 'Page-footer', 'id': 5, 'name': 'Page-footer'},
    {'supercategory': 'Page-header', 'id': 6, 'name': 'Page-header'},
    {'supercategory': 'Picture', 'id': 7, 'name': 'Picture'},
    {'supercategory': 'Section-header', 'id': 8, 'name': 'Section-header'},
    {'supercategory': 'Table', 'id': 9, 'name': 'Table'},
    {'supercategory': 'Text', 'id': 10, 'name': 'Text'},
    {'supercategory': 'Title', 'id': 11, 'name': 'Title'}
]

new_categories_pln4_dln4 = [
    {'supercategory': '', 'id': 1, 'name': 'text'},
    {'supercategory': '', 'id': 2, 'name': 'title'},
    {'supercategory': '', 'id': 3, 'name': 'table'},
    {'supercategory': '', 'id': 4, 'name': 'figure'}
]

category_mapping_pln4_dln4 = {
    1: 1,  # text
    2: 1,  # text
    3: 1,  # text
    4: 1,  # text
    # 5,6
    7: 4,  # figure
    8: 2,  # title
    9: 3,  # table
    10: 1,  # text
    11: 2  # title
}


source_basic_path = "/extern_home/cvhci/data/document_analysis/DocLayNet/COCO/"
target_basic_path = "/"


convert_dln11_to_dln4(source_basic_path=source_basic_path, target_basic_path=target_basic_path, split="val", new_categories_pln4_dln4=new_categories_pln4_dln4, category_mapping_pln4_dln4=category_mapping_pln4_dln4)
convert_dln11_to_dln4(source_basic_path=source_basic_path, target_basic_path=target_basic_path, split="test", new_categories_pln4_dln4=new_categories_pln4_dln4, category_mapping_pln4_dln4=category_mapping_pln4_dln4)
convert_dln11_to_dln4(source_basic_path=source_basic_path, target_basic_path=target_basic_path, split="train", new_categories_pln4_dln4=new_categories_pln4_dln4, category_mapping_pln4_dln4=category_mapping_pln4_dln4)


