"""
File: dla-sfda/datasets/convert_dataset_m6doc10.py
Description: Create Dataset M6Doc-10
"""

import json
from typing import List


def build_mapping_dict(class_mapping_M6DOC_to_DLN: dict, old_categories_m6doc: List[dict]) -> dict:
    """
    This function builds the mapping dict from m6doc to m6doc-10 in class id numbers
    Args:
        class_mapping_M6DOC_to_DLN: mapping dict
        old_categories_m6doc: old categories of m6doc in ms coco format

    Returns: dict with mapping id numbers
    """
    mapping_dict_list = []
    category_mapping_m6doc_m6doc10 = {}
    for value in old_categories_m6doc:
        old_class_name = value['name']
        old_class_id = value['id']

        new_class_name = class_mapping_M6DOC_to_DLN[old_class_name]
        new_class_id = [new_cat['id'] for new_cat in new_categories_m6doc_m6doc10 if new_cat['name'] == new_class_name]

        if len(new_class_id) > 0:
            mapping_dict_list.append(
                {"old_class_name": old_class_name,
                 "old_class_id": old_class_id,
                 "new_class_name": new_class_name,
                 "new_class_id": new_class_id[0]
                 }
            )
            category_mapping_m6doc_m6doc10[old_class_id] = new_class_id[0]

    return category_mapping_m6doc_m6doc10


def convert_m6doc_to_m6doc10(source_basic_path: str, target_basic_path: str, split: str, new_categories_m6doc_m6doc10: List[dict], category_mapping_m6doc_m6doc10: dict) -> None:
    """
    This function converts the M6Doc Dataset to the version with 10 classes (M6Doc-10)
    Args:
        source_basic_path: source path of your M6Doc Annotation Json
        target_basic_path: target path of the converted Annotation Json
        split: train, val, test
        new_categories_dln10_m6doc10: category list_dict of MS COCO
        category_mapping_m6doc_m6doc10: mapping dict
    """
    file_path=f'{source_basic_path}instances_{split}2017.json'
    with open(file_path, 'r') as file:
        original_annotation_COCO = json.load(file)
    print(f"Read {file_path}")

    original_annotation_COCO["categories"] = new_categories_m6doc_m6doc10
    original_annotation_COCO.pop('info')
    original_annotation_COCO.pop('licenses')
    original_annotation_COCO.pop('type')

    filtered_annotations = []
    for ann_element in original_annotation_COCO['annotations']:
        original_id = ann_element['category_id']
        if original_id in category_mapping_m6doc_m6doc10:
            ann_element['category_id'] = category_mapping_m6doc_m6doc10[original_id]
            filtered_annotations.append(ann_element)


    original_annotation_COCO['annotations'] = filtered_annotations
    save_path = f"{target_basic_path}M6Doc_with_10_Classes_{split}.json"
    with open(save_path, "w") as outfile:
        json.dump(original_annotation_COCO, outfile)
    print(f"Saved {save_path}")


# Table 4 of M6DOC Paper
class_mapping_M6DOC_to_DLN = {
    '_background_': '', # no class
    'QR code': '',
    'institute': 'Text',
    'advertisement': 'Picture',
    'jump line': 'Text',
    'algorithm': '',
    'kicker': 'Text',
    'answer': '',
    'lead': 'Text',
    'author': 'Text',
    'marginal note': 'Page-header',
    'barcode': '',
    'matching': '',
    'bill': '',
    'mugshot': 'Picture',
    'blank': '',
    'option': '',
    'bracket': '',
    'ordered list': '',
    'breakout': 'Text',
    'other question number': '',
    'byline': 'Text',
    'page number': 'Text',
    'caption': 'Caption',
    'paragraph': 'Text',
    'catalogue': '',
    'part': 'Title',
    'chapter title': 'Title',
    'play': '',
    'code': '',
    'poem': '',
    'correction': '',
    'reference': '',
    'credit': 'Text',
    'sealing line': '',
    'dateline': 'Text',
    'second-level question number': '',
    'drop cap': '',
    'second-level title': 'Title',
    "editor's note": 'Text',
    'section': 'Text',
    'endnote': 'Text',
    'section title': 'Title',
    'examinee information': '',
    'sidebar': '',
    'fifth-level title': 'Title',
    'sub section title': 'Title',
    'figure': 'Picture',
    'subhead': 'Title',
    'first-level question number': '',
    'subsub section title': 'Title',
    'first-level title': 'Title',
    'supplementary note': '',
    'flag': '',
    'table': 'Table',
    'folio': 'Section-header',
    'table caption': 'Caption',
    'footer': 'Page-footer',
    'table note': '',
    'footnote': 'Footnote',
    'teasers': '',
    'formula': 'Formula',
    'third-level question number': '',
    'fourth-level section title': 'Title',
    'third-level title': 'Title',
    'fourth-level title': 'Title',
    'title': 'Title',
    'header': 'Section-header',
    'translator': 'Text',
    'headline': 'Title',
    'underscore': '',
    'index': 'Page-header',
    'unordered list': '',
    'inside': '',
    'weather forecast': '',
}

old_categories_m6doc = [
     {'supercategory': None, 'id': 0, 'name': '_background_'},
     {'supercategory': None, 'id': 1, 'name': 'QR code'},
     {'supercategory': None, 'id': 2, 'name': 'advertisement'},
     {'supercategory': None, 'id': 3, 'name': 'algorithm'},
     {'supercategory': None, 'id': 4, 'name': 'answer'},
     {'supercategory': None, 'id': 5, 'name': 'author'},
     {'supercategory': None, 'id': 6, 'name': 'barcode'},
     {'supercategory': None, 'id': 7, 'name': 'bill'},
     {'supercategory': None, 'id': 8, 'name': 'blank'},
     {'supercategory': None, 'id': 9, 'name': 'bracket'},
     {'supercategory': None, 'id': 10, 'name': 'breakout'},
     {'supercategory': None, 'id': 11, 'name': 'byline'},
     {'supercategory': None, 'id': 12, 'name': 'caption'},
     {'supercategory': None, 'id': 13, 'name': 'catalogue'},
     {'supercategory': None, 'id': 14, 'name': 'chapter title'},
     {'supercategory': None, 'id': 15, 'name': 'code'},
     {'supercategory': None, 'id': 16, 'name': 'correction'},
     {'supercategory': None, 'id': 17, 'name': 'credit'},
     {'supercategory': None, 'id': 18, 'name': 'dateline'},
     {'supercategory': None, 'id': 19, 'name': 'drop cap'},
     {'supercategory': None, 'id': 20, 'name': "editor's note"},
     {'supercategory': None, 'id': 21, 'name': 'endnote'},
     {'supercategory': None, 'id': 22, 'name': 'examinee information'},
     {'supercategory': None, 'id': 23, 'name': 'fifth-level title'},
     {'supercategory': None, 'id': 24, 'name': 'figure'},
     {'supercategory': None, 'id': 25, 'name': 'first-level question number'},
     {'supercategory': None, 'id': 26, 'name': 'first-level title'},
     {'supercategory': None, 'id': 27, 'name': 'flag'},
     {'supercategory': None, 'id': 28, 'name': 'folio'},
     {'supercategory': None, 'id': 29, 'name': 'footer'},
     {'supercategory': None, 'id': 30, 'name': 'footnote'},
     {'supercategory': None, 'id': 31, 'name': 'formula'},
     {'supercategory': None, 'id': 32, 'name': 'fourth-level section title'},
     {'supercategory': None, 'id': 33, 'name': 'fourth-level title'},
     {'supercategory': None, 'id': 34, 'name': 'header'},
     {'supercategory': None, 'id': 35, 'name': 'headline'},
     {'supercategory': None, 'id': 36, 'name': 'index'},
     {'supercategory': None, 'id': 37, 'name': 'inside'},
     {'supercategory': None, 'id': 38, 'name': 'institute'},
     {'supercategory': None, 'id': 39, 'name': 'jump line'},
     {'supercategory': None, 'id': 40, 'name': 'kicker'},
     {'supercategory': None, 'id': 41, 'name': 'lead'},
     {'supercategory': None, 'id': 42, 'name': 'marginal note'},
     {'supercategory': None, 'id': 43, 'name': 'matching'},
     {'supercategory': None, 'id': 44, 'name': 'mugshot'},
     {'supercategory': None, 'id': 45, 'name': 'option'},
     {'supercategory': None, 'id': 46, 'name': 'ordered list'},
     {'supercategory': None, 'id': 47, 'name': 'other question number'},
     {'supercategory': None, 'id': 48, 'name': 'page number'},
     {'supercategory': None, 'id': 49, 'name': 'paragraph'},
     {'supercategory': None, 'id': 50, 'name': 'part'},
     {'supercategory': None, 'id': 51, 'name': 'play'},
     {'supercategory': None, 'id': 52, 'name': 'poem'},
     {'supercategory': None, 'id': 53, 'name': 'reference'},
     {'supercategory': None, 'id': 54, 'name': 'sealing line'},
     {'supercategory': None, 'id': 55, 'name': 'second-level question number'},
     {'supercategory': None, 'id': 56, 'name': 'second-level title'},
     {'supercategory': None, 'id': 57, 'name': 'section'},
     {'supercategory': None, 'id': 58, 'name': 'section title'},
     {'supercategory': None, 'id': 59, 'name': 'sidebar'},
     {'supercategory': None, 'id': 60, 'name': 'sub section title'},
     {'supercategory': None, 'id': 61, 'name': 'subhead'},
     {'supercategory': None, 'id': 62, 'name': 'subsub section title'},
     {'supercategory': None, 'id': 63, 'name': 'supplementary note'},
     {'supercategory': None, 'id': 64, 'name': 'table'},
     {'supercategory': None, 'id': 65, 'name': 'table caption'},
     {'supercategory': None, 'id': 66, 'name': 'table note'},
     {'supercategory': None, 'id': 67, 'name': 'teasers'},
     {'supercategory': None, 'id': 68, 'name': 'third-level question number'},
     {'supercategory': None, 'id': 69, 'name': 'third-level title'},
     {'supercategory': None, 'id': 70, 'name': 'title'},
     {'supercategory': None, 'id': 71, 'name': 'translator'},
     {'supercategory': None, 'id': 72, 'name': 'underscore'},
     {'supercategory': None, 'id': 73, 'name': 'unordered list'},
     {'supercategory': None, 'id': 74, 'name': 'weather forecast'}
]



new_categories_m6doc_m6doc10 = [
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


source_basic_path = "/extern_home/cvhci/data/document_analysis/M6Doc/annotations/"
target_basic_path = "/"

category_mapping_m6doc_m6doc10 = build_mapping_dict(class_mapping_M6DOC_to_DLN=class_mapping_M6DOC_to_DLN, old_categories_m6doc=old_categories_m6doc)
convert_m6doc_to_m6doc10(source_basic_path=source_basic_path, target_basic_path=target_basic_path, split="val", new_categories_m6doc_m6doc10=new_categories_m6doc_m6doc10, category_mapping_m6doc_m6doc10=category_mapping_m6doc_m6doc10)
convert_m6doc_to_m6doc10(source_basic_path=source_basic_path, target_basic_path=target_basic_path, split="test", new_categories_m6doc_m6doc10=new_categories_m6doc_m6doc10, category_mapping_m6doc_m6doc10=category_mapping_m6doc_m6doc10)
convert_m6doc_to_m6doc10(source_basic_path=source_basic_path, target_basic_path=target_basic_path, split="train", new_categories_m6doc_m6doc10=new_categories_m6doc_m6doc10, category_mapping_m6doc_m6doc10=category_mapping_m6doc_m6doc10)

