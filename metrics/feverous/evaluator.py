import numpy as np


class EvaluateTool(object):

    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):
        summary = {}
        all_match = []

        for pred, gold_item in zip(preds, golds):
            match_or_not = pred == gold_item['seq_out']
            all_match.append(match_or_not)

        summary["all"] = float(np.mean(all_match))

        return summary
