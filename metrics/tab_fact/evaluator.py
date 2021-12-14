# encoding=utf8
import numpy as np


class EvaluateTool(object):

    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):
        summary = {}
        all_match = []
        simple_match = []
        complex_match = []
        small_test_match = []

        for pred, gold_item in zip(preds, golds):
            match_or_not = pred == gold_item['seq_out']
            all_match.append(match_or_not)
            if gold_item["hardness"] == "simple":
                simple_match.append(match_or_not)
            else:
                complex_match.append(match_or_not)

            if gold_item["small_test"]:
                small_test_match.append(match_or_not)

        summary["all"] = float(np.mean(all_match))

        # TODO commented by Tianbao: for now we don't use too many metrics, only use "all" as indicator.
        # summary["simple"] = float(np.mean(simple_match))
        # summary["complex"] = float(np.mean(complex_match))
        # summary["small_test"] = float(np.mean(small_test_match))

        return summary
