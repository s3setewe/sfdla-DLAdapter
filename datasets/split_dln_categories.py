"""
File: dla-sfda/datasets/split_dln_categories.py
Description: Split Document Categories of DocLayNet Dataset
"""

import json


def run_splitting(split: str, source_basic_path: str, target_basic_path: str) -> None:
    """
    This function splits the Categories of the DocLayNet Dataset
    Args:
        split: train, val, test
        source_basic_path: source path of your DLN Annotation Json
        target_basic_path: target path of the splitted Annotation Json
    """
    print(split)
    file_path = f'{source_basic_path}{split}.json'
    with open(file_path, 'r') as f:
        json_data = json.load(f)

    new_categories = json_data['categories']

    for doc_category in set([items['doc_category'] for items in json_data['images']]):
        print(f"\t{doc_category}")
        new_images = [items for items in json_data['images'] if items['doc_category'] == doc_category]
        list_of_image_ids = [items['id'] for items in new_images]
        new_annotations = [items for items in json_data['annotations'] if items['image_id'] in list_of_image_ids]

        new_dict = {
            'images': new_images,
            'annotations': new_annotations,
            'categories': new_categories
        }

        with open(f'{target_basic_path}{doc_category}_{split}.json', 'w') as json_file:
            json.dump(new_dict, json_file, indent=4)


# Replace source_basic_path and target_basic_path with your path
source_basic_path = '/extern_home/cvhci/data/document_analysis/DocLayNet/COCO/'
target_basic_path = '/'


run_splitting(split="val", source_basic_path=source_basic_path, target_basic_path=target_basic_path)
run_splitting(split="test", source_basic_path=source_basic_path, target_basic_path=target_basic_path)
run_splitting(split="train", source_basic_path=source_basic_path, target_basic_path=target_basic_path)
