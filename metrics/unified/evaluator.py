# encoding=utf8
import re

import numpy as np


def maybe_normalize_float(span):
    if span and (re.match(r"^[+-][0-9]+[.]?[0-9]*$", span)
                 or (re.match(r"^[0-9]*[.]?[0-9]*$", span))) and span != '.':
        # FIXME: We did this(instead of try except) to convert a string into a float
        #  since the try catch will lead to an error when using 8 V100 gpus with cuda 11.0,
        #  and we still don't know why that could happen....
        return str(float(span))
    else:
        return span


def eval_ex_match(pred, gold_result):
    pred = [span.strip() for span in pred.split(', ')]
    gold_result = [span.strip() for span in gold_result.split(', ')]

    clean_float = True  # TODO
    if clean_float:
        pred = [maybe_normalize_float(span) for span in pred]
        gold_result = [maybe_normalize_float(span) for span in gold_result]

    return sorted(pred) == sorted(gold_result)


class EvaluateTool(object):

    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):
        summary = {}
        gold_inferreds = [item["seq_out"] for item in golds]

        ex_match = []
        for pred, gold_result in zip(preds, gold_inferreds):
            ex_match.append(eval_ex_match(pred, gold_result))

        summary["all_ex"] = float(np.mean(ex_match))
        return summary
