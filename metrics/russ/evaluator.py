def clean(parse):
    return " ".join(parse.strip().split()).lower()


class EvaluateTool(object):

    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):
        summary = {}
        correct = 0
        total = 0
        for pred, gold in zip(preds, golds):
            if clean(pred) == clean(gold["query"]):
                correct += 1
            total += 1
        summary["match"] = correct / total
        return summary
