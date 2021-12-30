# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" response entity hit metric. """
from typing import Optional, Union

import datasets

# TODO: contribute to the huggingface/datasets/metric
# TODO: format into the framework

_DESCRIPTION = """
The response entities hit is wrote or the task-oriented dialogue system tasks like KVRET
 and Multi-WoZ series datasets, since they use the F1 value got form comparison in entities
  between the prediction response and gold response.
  Actually, our evaluation logic is slightly different from those in Mem2seq, GLMP and DF-Net, 
  which makes it a bit lower in mark, but better reflect the real scenario of application.
"""

_KWARGS_DESCRIPTION = """
"""

_CITATION = """
Adopt from it, but slightly different.
@inproceedings{wu2019global,
  title={Global-to-local Memory Pointer Networks for Task-Oriented Dialogue},
  author={Wu, Chien-Sheng and Socher, Richard and Xiong, Caiming},
  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2019}
}
"""


def extract_entities_from_utterance(utterance, global_entities):
    """
    For seq2seq model
    @param utterance:
    @param global_entities:
    @return:
    """
    # Extract entities from utterance.
    def check_sub_str(str_list: list, sub_str: str):
        for str_item in str_list:
            if sub_str in str_item or sub_str.lower() in str_item.lower():
                return True
        return False

    utterance = " {} ".format(utterance)  # for entity matching
    for h in range(0, 13): # for formulating am & pm
        utterance = utterance.replace("{} am".format(h), "{}am".format(h))
        utterance = utterance.replace("{} pm".format(h), "{}pm".format(h))
    for entity_item_a in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
        for entity_item_b in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
            utterance = utterance.replace("{}-{}f".format(str(entity_item_a), str(entity_item_b)), "{}f-{}f".format(str(entity_item_a), str(entity_item_b)))
    entities_in_this_utterance = []
    entities = sorted(list(set(list(global_entities.values()) + list(global_entities.keys()))), key=lambda i: len(i), reverse=True)
    for entity in entities:
        if (entity in utterance) or (entity.lower() in utterance.lower()):
            if not check_sub_str(entities_in_this_utterance, entity):
                # in case of "week & weekend", "week & next_week" etc
                entities_in_this_utterance.append(entity.replace(' ', '_'))
    return list(set(entities_in_this_utterance))


def f1_score(y_pred, y_true, average="micro"):
    """
    F1 score for multi-label task
    @param y_pred: a list of list, each list item is a multi-label prediction
    @param y_true: a list of list, each list item is a multi-label gold result
    @param average: the average method, only micro and macro are supported
    @return:
    """
    assert len(y_pred) == len(y_true)

    def _compute_prf(gold, pred):
        TP, FP, FN = 0, 0, 0
        if len(gold) != 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in set(pred):
                if p not in gold:
                    FP += 1
            precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
            recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        else:
            precision, recall, F1, count = 0, 0, 0, 0
        return TP, FP, FN, F1, count

    def _compute_F1(precision, recall):
        F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        return F1

    F1_pred, F1_count, TP_all, FP_all, FN_all = 0, 0, 0, 0, 0

    for y_true_item, y_pred_item in zip(y_true, y_pred):
        single_tp, single_fp, single_fn, single_f1, count = _compute_prf(y_true_item, y_pred_item)
        F1_pred += single_f1
        F1_count += count
        TP_all += single_tp
        FP_all += single_fp
        FN_all += single_fn

    if average == "macro":
        F1_macro_score = F1_pred / float(F1_count)
        return F1_macro_score
    elif average == "micro":
        P_score = TP_all / float(TP_all + FP_all) if (TP_all + FP_all) != 0 else 0
        R_score = TP_all / float(TP_all + FN_all) if (TP_all + FN_all) != 0 else 0
        F1_micro_score = _compute_F1(P_score, R_score)
        return F1_micro_score
    else:
        raise ValueError("Options other than micro/macro are not supported.")


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Response_entity_hit(datasets.Metric):
    def __init__(
            self,
            config_name: Optional[str] = None,
            keep_in_memory: bool = False,
            cache_dir: Optional[str] = None,
            num_process: int = 1,
            process_id: int = 0,
            seed: Optional[int] = None,
            experiment_id: Optional[str] = None,
            max_concurrent_cache_files: int = 10000,
            timeout: Union[int, float] = 100,
            **kwargs
    ):
        super().__init__(
            config_name=config_name,
            keep_in_memory=keep_in_memory,
            cache_dir=cache_dir,
            num_process=num_process,
            process_id=process_id,
            seed=seed,
            experiment_id=experiment_id,
            max_concurrent_cache_files=max_concurrent_cache_files,
            timeout=timeout,
            **kwargs
        )

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references":
                        {
                            "response": datasets.Value("string"),  # gold response in this turn
                            "intents": datasets.Sequence(datasets.Value("string")),
                            # intent of the dialogue which this turn belongs to
                        }
                }
            )
        )

    # refer to GLMP and DF-Net https://github.com/LooperXX/DF-Net
    def _compute(self, *, predictions=None, references=None, **kwargs) -> Optional[dict]:
        """

        @param predictions: the predictions of model response to the history and knowledge base.
        @param references: the gold response of dialogue in the corresponding position.
        @return: the Navigate F1, Weather F1, Calendar F1 for KVRET dataset
        and Restaurant F1, Attraction F1, Hotel F1 for MultiWOZ 2.1
        """
        F1_scores = {}
        entities_from_predictions_and_references = {'all': {"predictions_entities": [], "references_entities": []}}
        global_entities = kwargs['global_entities']

        for prediction, response in zip(predictions, references):
            intents = response['intents']
            prediction_entities = extract_entities_from_utterance(utterance=prediction, global_entities=global_entities)
            references_entities = extract_entities_from_utterance(utterance=response['response'],
                                                                  global_entities=global_entities)
            entities_from_predictions_and_references["all"]['predictions_entities'].append(prediction_entities)
            entities_from_predictions_and_references["all"]['references_entities'].append(references_entities)
            for intent in intents:
                if intent not in entities_from_predictions_and_references.keys():
                    entities_from_predictions_and_references[intent] = {"predictions_entities": [],
                                                                        "references_entities": []}
                entities_from_predictions_and_references[intent]['predictions_entities'].append(prediction_entities)
                entities_from_predictions_and_references[intent]['references_entities'].append(references_entities)

        for category in entities_from_predictions_and_references.keys():
            prediction_entities = entities_from_predictions_and_references[category]['predictions_entities']
            references_entities = entities_from_predictions_and_references[category]['references_entities']
            F1_scores["{}_micro".format(category)] = f1_score(y_pred=prediction_entities, y_true=references_entities,
                                                              average="micro")
            F1_scores["{}_macro".format(category)] = f1_score(y_pred=prediction_entities, y_true=references_entities,
                                                              average="macro")

        return {**F1_scores}
