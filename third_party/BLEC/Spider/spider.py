from ..blec import BLECMetrics
from .eval import question_test
import json


class BLECSpider(BLECMetrics):
    def __init__(self, template_path=None, translate=False):
        super().__init__('spider')
        self.translate = translate
        if template_path is None:
            self.template_to_names = None
        else:
            self.template_to_names = self.load_template(template_path)

    def load_template(self, path):
        sql_to_names = json.load(open(path, 'r'))
        origin_sql_to_names = sql_to_names[0]
        trans_sql_to_names = sql_to_names[1]
        templates_to_names = origin_sql_to_names if not self.translate else trans_sql_to_names
        return templates_to_names

    def evaluate(self, pred, logic, gold=None):
        labels = []
        if self.template_to_names is None:
            raise NotImplementedError
        if logic not in self.template_to_names:
            print("Error: the template not found, the logic is ", logic)
        else:
            name_dict = self.template_to_names[logic]
            pred = pred.lower()
            _, labels = question_test(logic, name_dict, pred)
        return labels
