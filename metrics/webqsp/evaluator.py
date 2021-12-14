# encoding=utf8
from collections import defaultdict
from metrics.webqsp.utils import *

from rdflib.graph import Graph

kg_files = [f"third_party/webqsp/freebase150k_part{i}.txt" for i in range(3)]
kg_str = "".join([open(f).read() for f in kg_files])

g = Graph()
g.parse(data=kg_str, format="nt")


def execute_sparql(sparql):
    try:
        qres = g.query(sparql)
        answers = [str(a[0]) for a in qres]
        return answers
    except:
        return []


def compute_answers_F1(pred, gold):
    ans_ents = [e[0] for e in gold['answers']]
    try:
        sparql = lisp_to_sparql(pred)
        pred_ents = execute_sparql(sparql)
    except:
        pred_ents = []
    tp = len([p for p in pred_ents if p in ans_ents])
    P = tp / len(pred_ents) if len(pred_ents) else 0
    R = tp / len(ans_ents)  if len(ans_ents)  else 0
    F1 = 2 * (P * R) / (P + R) if (P + R) else 0
    return F1



class EvaluateTool(object):
    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):
        eval_dict = defaultdict(float)
        for pred, gold in zip(preds, golds):
            eval_dict["F1"] += compute_answers_F1(pred, gold)
        for key in eval_dict:
            # print (key, eval_dict[key], '/', len(golds))
            eval_dict[key] = eval_dict[key] / len(golds) if len(golds) else 0
        return eval_dict
