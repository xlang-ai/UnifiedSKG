# encoding=utf8
from third_party.BLEC.Logic2text import BLECLogic2text


class EvaluateTool(object):
    def __init__(self, args):
        self.args = args
        self.blec = BLECLogic2text()

    def evaluate(self, preds, golds, section):
        summary = {}

        logic_strs = [item["text_in"] for item in golds]
        gold_texts = [item["seq_out"] for item in golds]

        assert len(preds) == len(logic_strs) == len(gold_texts)
        true = []
        for logic_str, pred, gold_text in zip(logic_strs, preds, gold_texts):
            if len(self.blec.evaluate(logic_str, pred, gold_text)) == 0:
                true.append(1)
            else:
                true.append(0)

        summary["blec"] = 1.0 * sum(true) / len(preds)
        return summary
