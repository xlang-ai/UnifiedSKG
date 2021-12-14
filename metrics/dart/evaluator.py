# encoding=utf8

# from third_party.dart.evaluation.extract_scores_dart import extract_scores_dart
from datasets import load_metric


#  some of the code below refers to the https://github.com/Yale-LILY/FeTaQA/blob/main/end2end/train.py
def postprocess_text(preds, references_s, metric_name):
    preds = [pred.strip() for pred in preds]
    references_s = [[reference.strip() for reference in references] for references in references_s]

    if metric_name in ["metrics/dart/moverscore.py", "meteor", "bleurt"]:
        references_s = [references[0] for references in references_s]
    elif metric_name in ["sacrebleu"]:
        # since hf sacrebleu only support references with same length, we have to pad them into the same length
        ref_max_len = max([len(ref) for ref in references_s])
        # see https://github.com/mjpost/sacrebleu/pull/132
        references_s = [references + [None] * (ref_max_len - len(references)) for references in references_s]
    else:
        pass

    return preds, references_s


class EvaluateTool(object):
    def __init__(self, args):
        self.args = args
        # DART metrics: BLEU METEOR	TER	MoverScore BERTScore BLEURT PARENT

    def evaluate(self, preds, golds, section):
        summary = {}
        references_s = [item["references"] for item in golds]

        assert len(preds) == len(references_s)

        metric_list = []
        if section in ['train', 'dev']:
            metric_list = ["sacrebleu"]
        elif section == 'test':
            metric_list = ["sacrebleu", "meteor", "metrics/dart/ter.py",
            # "metrics/dart/moverscore.py"
            # # moverscore strictly depends on hf==3.1.0 which conflicts our repo dependency.
                                                  "bertscore", "bleurt"]

        # TODO: parent is not handy in Python, we will leave it in the final eval.
        for metric_name in metric_list:
            metric = load_metric(metric_name)
            processed_preds, processed_golds = postprocess_text(preds, references_s, metric_name)

            if metric_name == "bertscore":
                res = metric.compute(predictions=processed_preds, references=processed_golds, lang="en")
                for k, v in res.items():
                    if k == "hashcode":
                        continue
                    summary[f"{metric_name}_{k}"] = round(1.0 * sum(v) / len(v), 2)

            else:
                res = metric.compute(predictions=processed_preds, references=processed_golds)
                if metric_name == "sacrebleu":
                    summary[metric_name] = res["score"] * 0.01
                elif metric_name == "bleurt":
                    summary["bleurt"] = round(1.0 * sum(res["scores"]) / len(res["scores"]), 2)
                elif metric_name == "metrics/dart/moverscore.py":
                    summary[metric_name] = res["moverscore"]
                elif metric_name == "metrics/dart/ter.py":
                    summary[metric_name] = res["ter"]
                else:
                    summary[metric_name] = res[metric_name]
        return summary

if __name__ == '__main__':
    import json

    with open("predictions_predict.json") as f:
        test_data = json.load(f)
    preds = [item['prediction'] for item in test_data]
    evaluator = EvaluateTool(args=None)
    score = evaluator.evaluate(preds, test_data, section="test")
    print(score)