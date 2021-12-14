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
""" MoverScore metric. """
import statistics
import sys
from typing import Optional, Union

import datasets
from moverscore_v2 import get_idf_dict, word_mover_score

# TODO: contribute to the huggingface/datasets/metric
# TODO: format into the framework

_DESCRIPTION = """
MoverScore (Zhao et.al, 2019) is a monolingual measure of evaluating the similarity between a sentence pair written in the same language. It achieves much higher correlation with human judgments than BLEU on machine translation, summarization and image captioning. By default, this repo uses English BERT to measure the similarity of a English sentence pair. For other languages, a multilingual BERT in moverscore_v2.py has to be specified.
See https://github.com/AIPHES/emnlp19-moverscore#readme for more information.
"""

_KWARGS_DESCRIPTION = """
MoverScore.
Args:
    `translations` (list of str): prediction/candidate sentences
    `references` (list of str): reference sentences
Returns:
    'moverscore': the moverscore
Examples:
    >>> translations = ["hello there", "general kenobi"]
    >>> references = ["hello there", "general kenobi"]
    >>> moverscore = datasets.load_metric("moverscore") # we will add it to hf soon
    >>> moverscore = moverscore.compute(translations=translations, references=references)
    >>> print(moverscore['moverscore'])
"""

_CITATION = """
@inproceedings{zhao2019moverscore,
  title = {MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance},
  month = {August},
  year = {2019},
  author = {Wei Zhao, Maxime Peyrard, Fei Liu, Yang Gao, Christian M. Meyer, Steffen Eger},
  address = {Hong Kong, China},
  publisher = {Association for Computational Linguistics},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Moverscore(datasets.Metric):
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
                    "translations": datasets.Value("string"),
                    "references": datasets.Value("string"),
                }
            )
        )

    def _compute(self, *, translations=None, references=None, **kwargs) -> Optional[dict]:
        idf_dict_hyp = get_idf_dict(translations)  # idf_dict_hyp = defaultdict(lambda: 1.)#translations is a list of candidate sentences
        idf_dict_ref = get_idf_dict(references)  # idf_dict_ref = defaultdict(lambda: 1.)
        # reference is a list of reference sentences
        scores = word_mover_score(references, translations, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1,
                                  remove_subwords=True)

        return {"moverscore": statistics.mean(scores)}
