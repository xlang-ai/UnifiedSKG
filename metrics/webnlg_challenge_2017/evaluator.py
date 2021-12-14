# encoding=utf8
import os

from third_party.dart import extract_score_webnlg


def evaluate_webnlg_challenge_2017(references_s, preds):
    """
    The evaluation of the webnlg_challenge_2017,
    we use the evaluate shell that DART dataset provided.
    :param references_s: ACTUALLY, references in webnlg are of no use.
    :param preds:
    :return:
    """

    tmp_file_name = 'webnlg_challenge_2017_tmp4eval.txt'
    with open(tmp_file_name, 'w') as tmp_file:
        for pred in preds:
            print(pred, file=tmp_file)

    os.system("bash utils/process/general/dart_lib/run_eval_on_webnlg.sh "
              "{}".format(tmp_file_name))

    summary = extract_score_webnlg()

    return summary


class EvaluateTool(object):
    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):
        references_s = [item["references"] for item in golds]

        assert len(preds) == len(references_s)

        summary = evaluate_webnlg_challenge_2017(references_s, preds)

        return summary
