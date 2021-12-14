# encoding=utf8
from third_party.BLEC.Spider import BLECSpider


class EvaluateTool(object):

    def __init__(self, args):
        self.args = args
        self.blec_test = BLECSpider(template_path="third_party/BLEC/template_to_names_test.json")
        self.blec_dev = BLECSpider(template_path="third_party/BLEC/template_to_names_dev.json")

    def evaluate(self, preds, golds, section):
        if section == 'test':
            blec = self.blec_test
        elif section == 'dev':
            blec = self.blec_dev
        else:
            raise ValueError()

        summary = {}

        sqls = [item["text_in"] for item in golds]

        assert len(preds) == len(sqls)
        true = []
        for sql, pred in zip(sqls, preds):

            if len(blec.evaluate(pred, sql)) == 0:
                true.append(1)
            else:
                true.append(0)

        summary["blec"] = 1.0 * sum(true) / len(preds)
        return summary
