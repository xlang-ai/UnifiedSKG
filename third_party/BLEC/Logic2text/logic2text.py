from ..blec import BLECMetrics
from .eval import logic_matching


class BLECLogic2text(BLECMetrics):
    def __init__(self):
        super().__init__('logic2text')

    def evaluate(self, pred, logic, gold):
        labels = logic_matching(pred, logic, gold)
        return labels
