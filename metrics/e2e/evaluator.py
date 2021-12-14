# encoding=utf8

from third_party.e2e.measure_scores import evaluate_e2e


class EvaluateTool(object):
    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):
        references_s = [item["references"] for item in golds]
        assert len(preds) == len(references_s)

        summary = evaluate_e2e(references_s, preds)

        return summary
