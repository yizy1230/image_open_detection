# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from typing import List, Optional
import re
from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .base_det_dataset import BaseDetDataset


def remove_speculative_clauses(text):
    # Keywords that mark a guess or uncertainty
    speculative_keywords = [
        'possibly',
        'probable', 'probably',
        'maybe',
        'perhaps',
        'suggest', 'suggests', 'suggesting', 'suggested',
        'speculate', 'speculates', 'speculating', 'speculated',
        'assume', 'assumes', 'assuming', 'assumed',
        'guess', 'guesses', 'guessing', 'guessed',
        'suppose', 'supposes', 'supposing', 'supposed',
        'appear', 'appears', 'appeared', 'appearing',
        'seem', 'seems', 'seemed', 'seeming',
        'indicate', 'indicates', 'indicating', 'indicated',
        'hint', 'hints', 'hinting', 'hinted',
        'imply', 'implies', 'implying', 'implied',
        'infer', 'infers', 'inferring', 'inferred',
        'expect', 'expects', 'expecting', 'expected',
        'assuming that', 'tends to', 'tends not to', 'seems to', 'looks like', 
        'sounds like', 'appears to',
        ' no ', ' not '
    ]
    
    # Common prepositions
    prepositions = [
        'in', 'on', 'at', 'by', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
        'above', 'below', 'to', 'from', 'up', 'down', 'for', 'of', 'under', 'over', 'as'
    ]
    
    # Regex for clauses that contain a hedging keyword
    clause_pattern = r',?[^,]*\b(?:' + '|'.join(speculative_keywords) + r')\b[^,]*,?'
    
    # Drop comma-separated clauses that contain a hedging keyword
    def remove_speculative_subclauses(sentence):
        clean_sentence = re.sub(clause_pattern, '', sentence).strip()
        
        # Add a period if the sentence has no punctuation after the clause is removed
        if not clean_sentence.endswith('.'):
            clean_sentence += '.'
        
        return clean_sentence
    
    # If only a prepositional fragment remains, drop it
    def is_prepositional_phrase_only(sentence):
        # Split the sentence into words
        words = sentence.strip().split()
        # A sentence whose first word is a preposition is only a prepositional phrase
        if words and words[0].lower() in prepositions:
            return True
        return False
    
    # Split the text into sentences on periods
    sentences = re.split(r'(?<=\.)\s+', text)
    
    # For each sentence, drop a clause that contains a hedging keyword
    clean_sentences = []
    for sentence in sentences:
        if any(keyword in sentence for keyword in speculative_keywords):
            # If the sentence hedges, drop the clause
            clean_sentence = remove_speculative_subclauses(sentence)
            # If what remains starts with a preposition, drop the whole sentence
            if clean_sentence.strip() and not is_prepositional_phrase_only(clean_sentence):
                if clean_sentence.strip() != '.':
                    clean_sentences.append(clean_sentence.strip())
        else:
            clean_sentences.append(sentence.strip())
    
    # Return the cleaned text
    return ' '.join(clean_sentences).strip()




@DATASETS.register_module()
class ODVGDataset(BaseDetDataset):
    """object detection and visual grounding dataset."""

    def __init__(self,
                 *args,
                 data_root: str = '',
                 label_map_file: Optional[str] = None,
                 need_text: bool = True,
                 actual_dataset_mode='VG',
                 use_short_cap=False,
                 use_uniform_prompt=True,
                 clean_caption=True,
                 **kwargs) -> None:
        self.dataset_mode = 'VG'
        self.actual_dataset_mode = actual_dataset_mode
        self.need_text = need_text
        self.use_short_cap = use_short_cap
        self.use_uniform_prompt = use_uniform_prompt
        self.clean_caption = clean_caption
        if label_map_file:
            label_map_file = osp.join(data_root, label_map_file)
            with open(label_map_file, 'r') as file:
                self.label_map = json.load(file)
            self.dataset_mode = 'OD'
        super().__init__(*args, data_root=data_root, **kwargs)
        assert self.return_classes is True

    def load_data_list(self) -> List[dict]:
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            with open(local_path, 'r') as f:
                data_list = [json.loads(line) for line in f]

        out_data_list = []
        for data in data_list:
            data_info = {}
            img_path = osp.join(self.data_prefix['img'], data['filename'])
            data_info['img_path'] = img_path
            data_info['height'] = data['height']
            data_info['width'] = data['width']
            if 'conversations' in data.keys():
                if self.actual_dataset_mode == 'VG' and self.use_short_cap and 'tags' in data.keys():
                    subsentence = data['grounding']['caption'].split('.')
                    subsentence = [sen.strip() for sen in subsentence]
                    subsentence = [sen for sen in subsentence if len(sen) > 0]
                    subsentence = [sen for sen in subsentence if sen not in data['tags']]
                    data['conversations'][1]["value"] = '. '.join(subsentence) + '.'
                elif self.clean_caption:
                    new_conv = remove_speculative_clauses(data['conversations'][1]["value"])
                    data['conversations'][1]["value"] = new_conv if len(new_conv) else data['conversations'][1]["value"]
                if self.use_uniform_prompt:
                    data['conversations'][0]["value"] = '<image>\nDescribe the image in detail.'
                data_info['conversations'] = data['conversations']
            else:
                data_info['conversations'] = []
            if 'tags' in data.keys():
                data_info['tags'] = data['tags']
            else:
                data_info['tags'] = []
            if self.dataset_mode == 'OD':
                if self.need_text:
                    data_info['text'] = self.label_map
                anno = data.get('detection', {})
                instances = [obj for obj in anno.get('instances', [])]
                bboxes = [obj['bbox'] for obj in instances]
                bbox_labels = [str(obj['label']) for obj in instances]

                instances = []
                for bbox, label in zip(bboxes, bbox_labels):
                    instance = {}
                    x1, y1, x2, y2 = bbox
                    inter_w = max(0, min(x2, data['width']) - max(x1, 0))
                    inter_h = max(0, min(y2, data['height']) - max(y1, 0))
                    if inter_w * inter_h == 0:
                        continue
                    if (x2 - x1) < 1 or (y2 - y1) < 1:
                        continue
                    instance['ignore_flag'] = 0
                    instance['bbox'] = bbox
                    instance['bbox_label'] = int(label)
                    instances.append(instance)
                data_info['instances'] = instances
                data_info['dataset_mode'] = self.actual_dataset_mode
                out_data_list.append(data_info)
            else:
                anno = data['grounding']
                data_info['text'] = anno['caption']
                regions = anno['regions']

                instances = []
                phrases = {}
                for i, region in enumerate(regions):
                    bbox = region['bbox']
                    phrase = region['phrase']
                    tokens_positive = region['tokens_positive']
                    if not isinstance(bbox[0], list):
                        bbox = [bbox]
                    for box in bbox:
                        instance = {}
                        x1, y1, x2, y2 = box
                        inter_w = max(0, min(x2, data['width']) - max(x1, 0))
                        inter_h = max(0, min(y2, data['height']) - max(y1, 0))
                        if inter_w * inter_h == 0:
                            continue
                        if (x2 - x1) < 1 or (y2 - y1) < 1:
                            continue
                        instance['ignore_flag'] = 0
                        instance['bbox'] = box
                        instance['bbox_label'] = i
                        phrases[i] = {
                            'phrase': phrase,
                            'tokens_positive': tokens_positive
                        }
                        instances.append(instance)
                data_info['instances'] = instances
                data_info['phrases'] = phrases
                data_info['dataset_mode'] = self.actual_dataset_mode
                out_data_list.append(data_info)

        del data_list
        return out_data_list
    
    # def prepare_data(self, idx):
    #     """Get data processed by ``self.pipeline``.

    #     Args:
    #         idx (int): The index of ``data_info``.

    #     Returns:
    #         Any: Depends on ``self.pipeline``.
    #     """
    #     img_path = getattr(self, 'img_path', None)
    #     data = self.get_data_info(idx)
    #     if img_path == None:
    #         self.img_path = data['img_path']
    #     for t in self.pipeline.transforms:
    #         data = t(data)
    #         if data['img_path'] == self.img_path:
    #             print(t)
    #             print(data)
    #         # The transform will return None when it failed to load images or
    #         # cannot find suitable augmentation parameters to augment the data.
    #         # Here we simply return None if the transform returns None and the
    #         # dataset will handle it by randomly selecting another data sample.
    #         if data is None:
    #             return None
    #     return data
        # return self.pipeline(data_info)
