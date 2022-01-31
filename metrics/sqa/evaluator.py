# encoding=utf8

import numpy as np
from metrics.unified.evaluator import eval_ex_match


class EvaluateTool(object):

    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):
        summary = {}
        all_match = []
        interaction_match = {}
        pos_match_dic = {"0": [], "1": [], "2": [], "3": []}

        for pred, gold_item in zip(preds, golds):
            gold_seq_out = gold_item['seq_out']
            match_or_not = eval_ex_match(pred, gold_seq_out)

            # Add the match result to the all set.
            all_match.append(match_or_not)

            # Get the position tag.
            _pos = str(gold_item['position'])

            # Add the match result to the corresponding position set.
            if _pos in pos_match_dic.keys():
                pos_match_dic[_pos].append(match_or_not)
                # We only count acc in the top-4 question(pos 0, 1, 2, 3) and all(0,1,2,3,4...)

            # Get the id tag.
            sid = "{}\t{}".format(gold_item['id'], gold_item['annotator'])
            if sid not in interaction_match.keys():
                interaction_match[sid] = []
            interaction_match[sid].append(match_or_not)

        summary["all_acc"] = float(np.mean(all_match))
        for i in pos_match_dic.keys():
            summary["pos_{}_acc".format(i)] = float(np.mean(pos_match_dic[i]))
        summary["interaction_acc"] = float(
            np.mean([all(matches) for interaction_id, matches in interaction_match.items()]))

        return summary