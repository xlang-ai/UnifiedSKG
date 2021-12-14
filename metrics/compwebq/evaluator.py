# encoding=utf8
from collections import defaultdict

def compute_answers_hits_at_1(pred, gold):
    ans_ents = gold['answers']
    pred_ent = [p.strip() for p in pred.split(', ')][0]
    return (pred_ent in set(ans_ents))

def compute_answers_exact_match(pred, gold):
    ans_ents = gold['answers']
    pred_ents = [p.strip() for p in pred.split(', ')]
    return (set(pred_ents) == set(ans_ents))

def compute_answers_F1(pred, gold):
    ans_ents = gold['answers']
    pred_ents = [p.strip() for p in pred.split(', ')]
    tp = len([p for p in pred_ents if p in ans_ents])
    P = tp / len(pred_ents) if len(pred_ents) else 0
    R = tp / len(ans_ents)  if len(ans_ents)  else 0
    F1 = 2 * (P * R) / (P + R) if (P + R) else 0
    return F1

def compute_sparql_exact_match(pred, gold):
    sparql = gold['sparql']
    return (pred.strip() == sparql.strip())



class EvaluateTool(object):
    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):
        eval_dict = defaultdict(float)
        for pred, gold in zip(preds, golds):
            eval_dict["acc"] += compute_answers_exact_match(pred, gold)
            eval_dict["F1"] += compute_answers_F1(pred, gold)
            eval_dict["Hits@1"] += compute_answers_hits_at_1(pred, gold)
            #eval_dict["exact_match"] += compute_sparql_exact_match(pred, gold)
        for key in eval_dict:
            eval_dict[key] = eval_dict[key] / len(golds) if len(golds) else 0
        return eval_dict
