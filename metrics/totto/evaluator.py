# encoding=utf8
import numpy as np
from datasets import load_metric


#  the code below refers to the https://github.com/Yale-LILY/FeTaQA/blob/main/end2end/train.py
def postprocess_text(preds, references_s, metric_name):
    preds = [pred.strip() for pred in preds]
    references_s = [[reference.strip() for reference in references] for references in references_s]

    # rougeLSum expects newline after each sentence
    if metric_name in ["sacrebleu"]:
        # since hf sacrebleu only support references with same length, we have to pad them into the same length
        ref_max_len = max([len(ref) for ref in references_s])
        for ref in references_s:
            for _ in range(ref_max_len - len(ref)):
                ref.append(None)  # see https://github.com/mjpost/sacrebleu/pull/132
                print(ref)
    elif metric_name == "bleu":
        preds = [pred.split(' ') for pred in preds]
        references_s = [[reference.split(' ') for reference in references] for references in references_s]
    else:
        pass

    return preds, references_s


class EvaluateTool(object):
    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):
        summary = {}
        references_s = [item["final_sentences"] for item in golds]

        assert len(preds) == len(references_s)

        metric_list = []
        if section in ['train', 'dev']:
            metric_list = ['sacrebleu']
        elif section == 'test':
            metric_list = ["sacrebleu", "bleurt"]  # TODO: add PARENT

        for metric_name in metric_list:
            metric = load_metric(metric_name)
            processed_preds, processed_golds = postprocess_text(preds, references_s, metric_name)

            if metric_name == "sacrebleu":
                res = metric.compute(predictions=processed_preds, references=processed_golds)
                summary[metric_name] = res["score"] * 0.01
            elif metric_name == "bleurt":
                # We refer to the realization in https://github.com/google-research/language/blob/13fd14e1b285002412252097586f8fe405ba8a24/language/totto/totto_bleurt_eval.py#L94-L131
                multi_references = [[], [], []]
                for references in processed_golds:  # here "references" mean references for one prediction string.
                    if len(references) == 2:
                        multi_references[2].append('')
                    elif len(references) == 3:
                        multi_references[2].append(references[2])
                    else:
                        raise ValueError("The references num for each candidate should be 2 or 3 in ToTTo dataset.")
                    multi_references[0].append(references[0])
                    multi_references[1].append(references[1])

                multi_bleurt_scores = []
                for references in multi_references:
                    multi_bleurt_scores.append(metric.compute(predictions=processed_preds, references=references))

                assert len(multi_references) == 3
                avg_bleurt_scores = []
                for i in range(len(processed_preds)):
                    # All examples have atleast two references but some do not have three.
                    assert multi_references[0][i] and multi_references[1][i]
                    r2 = multi_references[2][i]
                    if r2:
                        # Take average over 3 references.
                        score_i = (multi_bleurt_scores[0][i] + multi_bleurt_scores[1][i] +
                                   multi_bleurt_scores[2][i]) / 3
                    else:
                        # print("only two refs")
                        # Take average over two references.
                        score_i = (multi_bleurt_scores[0][i] + multi_bleurt_scores[1][i]) / 2
                    avg_bleurt_scores.append(score_i)
                summary["bleurt"] = np.mean(avg_bleurt_scores)
            else:
                res = metric.compute(predictions=processed_preds, references=processed_golds)
                summary[metric_name] = res[metric_name]
        return summary


if __name__ == '__main__':
    import json

    with open("predictions_eval_3.179650238473768.json") as f:
        test_data = json.load(f)

    with open("dev_result.txt") as f:
        preds = [line.strip() for line in f.readlines()]

    evaluator = EvaluateTool(args=None)
    score = evaluator.evaluate(preds, test_data, section="test")
    print(score)
