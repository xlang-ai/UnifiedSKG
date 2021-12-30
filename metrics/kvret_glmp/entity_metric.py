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
import json
from typing import Optional, Union

import datasets

# TODO: contribute to the huggingface/datasets/metric
# TODO: format into the framework

_DESCRIPTION = """
The response entities hit is wrote or the task-oriented dialogue system tasks like KVRET
 and Multi-WoZ series datasets, since they use the F1 value got form comparison in entities
  between the prediction response and gold response.
"""

_KWARGS_DESCRIPTION = """
"""

_CITATION = """
Adopt from it.
@inproceedings{qin-etal-2020-dynamic,
    title = "Dynamic Fusion Network for Multi-Domain End-to-end Task-Oriented Dialog",
    author = "Qin, Libo  and
      Xu, Xiao  and
      Che, Wanxiang  and
      Zhang, Yue  and
      Liu, Ting",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.565",
    pages = "6344--6354",
    abstract = "Recent studies have shown remarkable success in end-to-end task-oriented dialog system. However, most neural models rely on large training data, which are only available for a certain number of task domains, such as navigation and scheduling. This makes it difficult to scalable for a new domain with limited labeled data. However, there has been relatively little research on how to effectively use data from all domains to improve the performance of each domain and also unseen domains. To this end, we investigate methods that can make explicit use of domain knowledge and introduce a shared-private network to learn shared and specific knowledge. In addition, we propose a novel Dynamic Fusion Network (DF-Net) which automatically exploit the relevance between the target domain and each domain. Results show that our models outperforms existing methods on multi-domain dialogue, giving the state-of-the-art in the literature. Besides, with little training data, we show its transferability by outperforming prior best model by 13.9{\%} on average.",
}
"""


def evaluate(preds, golds, entity_path):
    print("STARTING EVALUATION")
    acc, total = 0, 0
    domain2kvr_name_domain = {
        "all": 'ent_index',
        "calendar": 'ent_idx_cal',
        "navigate": 'ent_idx_nav',
        "weather": 'ent_idx_wet'
    }
    F1_pred = {"all": 0,
               "calendar": 0,
               "navigate": 0,
               "weather": 0}
    F1_count = {"all": 0,
                "calendar": 0,
                "navigate": 0,
                "weather": 0}
    TP = {"all": 0,
          "calendar": 0,
          "navigate": 0,
          "weather": 0}
    FP = {"all": 0,
          "calendar": 0,
          "navigate": 0,
          "weather": 0}
    FN = {"all": 0,
          "calendar": 0,
          "navigate": 0,
          "weather": 0}

    with open(entity_path) as f:
        global_entity = json.load(f)
        global_entity_type = {}
        global_entity_list = []
        for key in global_entity.keys():
            if key != 'poi':
                entity_arr = [item.lower().replace(' ', '_') for item in global_entity[key]]
                global_entity_list += entity_arr
                for entity in entity_arr:
                    global_entity_type[entity] = key
            else:
                for item in global_entity['poi']:
                    entity_arr = [item[k].lower().replace(' ', '_') for k in item.keys()]
                    global_entity_list += entity_arr
                    for key in item:
                        global_entity_type[item[key].lower().replace(' ', '_')] = key
        global_entity_list = list(set(global_entity_list))

    for pred, gold in zip(preds, golds):
        pred_sent = pred.lstrip().rstrip()
        gold_sent = gold['response'].lstrip().rstrip()
        for domain, kvr_name in domain2kvr_name_domain.items():
            # compute F1 SCORE
            single_tp, single_fp, single_fn, single_f1, count = compute_prf(gold[kvr_name],
                                                                            pred_sent.split(),
                                                                            global_entity_list,
                                                                            # gold['kb_arr_plain']
                                                                            gold['kb_arr']
                                                                            # kb_arr from gold isn't change, it is plain
                                                                            )
            F1_pred[domain] += single_f1
            F1_count[domain] += count
            TP[domain] += single_tp
            FP[domain] += single_fp
            FN[domain] += single_fn

        # compute Per-response Accuracy Score
        total += 1
        if (gold_sent == pred_sent):
            acc += 1

    acc_score = acc / float(total)
    summary = {}
    for domain in domain2kvr_name_domain.keys():
        summary["{}_F1_macro".format(domain)] = F1_pred[domain] / float(F1_count[domain])

    P_score = {}
    R_score = {}
    for domain in domain2kvr_name_domain.keys():
        P_score[domain] = TP[domain] / float(TP[domain] + FP[domain]) if (TP[domain] + FP[domain]) != 0 else 0
        R_score[domain] = TP[domain] / float(TP[domain] + FN[domain]) if (TP[domain] + FN[domain]) != 0 else 0
        summary["{}_F1_micro".format(domain)] = compute_F1(P_score[domain], R_score[domain])
    return summary


def compute_prf(gold, pred, global_entity_list, kb_plain):
    local_kb_word = [k[-1] for k in kb_plain] # since we didn't give reversed order in loader
    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in set(pred):
            if p in global_entity_list or p in local_kb_word:
                if p not in gold:
                    FP += 1
        precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
        recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
    else:
        precision, recall, F1, count = 0, 0, 0, 0
    return TP, FP, FN, F1, count


def compute_F1(precision, recall):
    F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
    return F1


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

                            "kb_arr": datasets.Sequence((datasets.Sequence(datasets.Value("string")))),
                            'response': datasets.Value("string"),
                            'ent_index': datasets.Sequence(datasets.Value("string")),
                            "ent_idx_cal": datasets.Sequence(datasets.Value("string")),
                            "ent_idx_wet": datasets.Sequence(datasets.Value("string")),
                            "ent_idx_nav": datasets.Sequence(datasets.Value("string")),
                            'domain': datasets.Value("string"),
                            "entities_file": datasets.Value("string"),

                        }
                }
            )
        )

    # refer to GLMP and DF-Net https://github.com/LooperXX/DF-Net
    def _compute(self, *, predictions=None, references=None, **kwargs) -> Optional[dict]:
        entities_path = references[0]['entities_file']
        return {**evaluate(preds=predictions, golds=references, entity_path=entities_path)}
