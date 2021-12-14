# encoding=utf8
from collections import defaultdict

import numpy as np
from metrics.unified.evaluator import eval_ex_match


# this function is adapt from tapex
'''def evaluate_example(_predict_str: str, _ground_str: str):
    target_delimiter = ', '
    _predict_spans = _predict_str.split(target_delimiter)
    _ground_spans = _ground_str.split(target_delimiter)
    _predict_values = defaultdict(lambda: 0)
    _ground_values = defaultdict(lambda: 0)
    for span in _predict_spans:
        # try:
        #     _predict_values[float(span)] += 1
        # except ValueError:
            _predict_values[span.strip()] += 1
    for span in _ground_spans:
        # try:
        #     _ground_values[float(span)] += 1
        # except ValueError:
            _ground_values[span.strip()] += 1
    _is_correct = _predict_values == _ground_values

    return _is_correct'''


class EvaluateTool(object):

    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):
        summary = {}
        gold_inferreds = [item["seq_out"] for item in golds]

        ex_match = []
        for pred, gold_result in zip(preds, gold_inferreds):
            ex_match.append(eval_ex_match(pred, gold_result))
            # ex_match.append(sorted(pred.split(', ')) == sorted(gold_result.split(', ')))

        summary["all_ex"] = float(np.mean(ex_match))
        return summary
