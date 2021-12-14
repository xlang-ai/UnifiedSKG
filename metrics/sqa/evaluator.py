# encoding=utf8
from collections import defaultdict

import numpy as np
from metrics.unified.evaluator import eval_ex_match

#
# # this function is adapt from tapex
# def evaluate_example(_predict_str: str, _ground_str: str):
#     target_delimiter = ', '
#     _predict_spans = _predict_str.split(target_delimiter)
#     _ground_spans = _ground_str.split(target_delimiter)
#     _predict_values = defaultdict(lambda: 0)
#     _ground_values = defaultdict(lambda: 0)
#     for span in _predict_spans:
#         # try:
#         #     _predict_values[float(span)] += 1
#         # except ValueError:
#             _predict_values[span.strip()] += 1
#     for span in _ground_spans:
#         # try:
#         #     _ground_values[float(span)] += 1
#         # except ValueError:
#             _ground_values[span.strip()] += 1
#     _is_correct = _predict_values == _ground_values
#
#     return _is_correct


class EvaluateTool(object):

    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):
        summary = {}
        all_match = []
        pos_match_dic = {"0": [], "1": [], "2": [], "3": []}

        for pred, gold_item in zip(preds, golds):
            gold_seq_out = gold_item['seq_out']
            match_or_not = eval_ex_match(pred, gold_seq_out)
            # Add the match result to the all set.
            all_match.append(match_or_not)

            # Add the match result to the corresponding position set.
            _pos = str(gold_item['position'])
            if _pos in pos_match_dic.keys():
                pos_match_dic[_pos].append(match_or_not)
                # We only count acc in the top-4 question(pos 0, 1, 2, 3) and all(0,1,2,3,4...)

        summary["all_acc"] = float(np.mean(all_match))
        for i in pos_match_dic.keys():
            summary["pos_{}_acc".format(i)] = float(np.mean(pos_match_dic[i]))

        return summary
