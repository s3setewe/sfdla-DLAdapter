"""
File: dla-sfda/datasets/convert_dataset_dln10.py
Description: Create Dataset DLN-10
"""

import json
from typing import List


def convert_dln11_to_dln10(source_basic_path: str, target_basic_path: str, split: str, new_categories_dln10_m6doc10: List[dict], category_mapping_dln11_dln10: dict) -> None:
    """
    This function converts the DocLayNet Dataset (11 Classes) to the version with 10 classes (DLN-10)
    Args:
        source_basic_path: source path of your DLN Annotation Json
        target_basic_path: target path of the converted Annotation Json
        split: train, val, test
        new_categories_dln10_m6doc10: category list_dict of MS COCO
        category_mapping_dln11_dln10: mapping dict
    """
    file_path = f'{source_basic_path}{split}.json'
    with open(file_path, 'r') as file:
        original_annotation_COCO = json.load(file)
    print(f"Read {file_path}")

    original_annotation_COCO["categories"] = new_categories_dln10_m6doc10
    filtered_annotations = []
    for ann_element in original_annotation_COCO['annotations']:
        original_id = ann_element['category_id']
        if original_id in category_mapping_dln11_dln10:
            ann_element['category_id'] = category_mapping_dln11_dln10[original_id]
            filtered_annotations.append(ann_element)
    original_annotation_COCO['annotations'] = filtered_annotations

    save_path = f"{target_basic_path}DLN_with_10_Classes_{split}.json"
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

new_categories_dln10_m6doc10 = [
    {'supercategory': None, 'id': 0, 'name': 'Caption'},
    {'supercategory': None, 'id': 1, 'name': 'Footnote'},
    {'supercategory': None, 'id': 2, 'name': 'Formula'},
    {'supercategory': None, 'id': 3, 'name': 'Page-footer'},
    {'supercategory': None, 'id': 4, 'name': 'Page-header'},
    {'supercategory': None, 'id': 5, 'name': 'Picture'},
    {'supercategory': None, 'id': 6, 'name': 'Section-header'},
    {'supercategory': None, 'id': 7, 'name': 'Table'},
    {'supercategory': None, 'id': 8, 'name': 'Text'},
    {'supercategory': None, 'id': 9, 'name': 'Title'}
]

category_mapping_dln11_dln10 = {
    1: 0,
    2: 1,
    3: 2,
    # 4
    5: 3,
    6: 4,
    7: 5,
    8: 6,
    9: 7,
    10: 8,
    11: 9
}

source_basic_path = "/extern_home/cvhci/data/document_analysis/DocLayNet/COCO/"
target_basic_path = "/"

convert_dln11_to_dln10(source_basic_path=source_basic_path, target_basic_path=target_basic_path, split="val", new_categories_dln10_m6doc10=new_categories_dln10_m6doc10, category_mapping_dln11_dln10=category_mapping_dln11_dln10)
convert_dln11_to_dln10(source_basic_path=source_basic_path, target_basic_path=target_basic_path, split="test", new_categories_dln10_m6doc10=new_categories_dln10_m6doc10, category_mapping_dln11_dln10=category_mapping_dln11_dln10)
convert_dln11_to_dln10(source_basic_path=source_basic_path, target_basic_path=target_basic_path, split="train", new_categories_dln10_m6doc10=new_categories_dln10_m6doc10, category_mapping_dln11_dln10=category_mapping_dln11_dln10)

