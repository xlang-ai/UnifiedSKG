# encoding=utf8
from collections import defaultdict
from third_party.grailqa.evaluator import *

fb_roles = "third_party/grailqa/ontology/fb_roles"
fb_types = "third_party/grailqa/ontology/fb_types"
reverse_properties = "third_party/grailqa/ontology/reverse_properties"


reverse_properties, relation_dr, relations, upper_types, types = process_ontology(fb_roles, fb_types, reverse_properties)
matcher = SemanticMatcher(reverse_properties, relation_dr, relations, upper_types, types)


def compute_exact_match(pred, gold):
    ref = gold['s_expression']
    em = int(matcher.same_logical_form(pred, ref))
    return em



class EvaluateTool(object):
    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):
        eval_dict = defaultdict(float)
        for pred, gold in zip(preds, golds):
            eval_dict["exact_match"] += compute_exact_match(pred, gold)
        for key in eval_dict:
            # print (key, eval_dict[key], '/', len(golds))
            eval_dict[key] = eval_dict[key] / len(golds) if len(golds) else 0
        return eval_dict
