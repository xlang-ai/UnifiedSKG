# encoding=utf8
from datasets import load_metric


#  the code below refers to the https://github.com/Yale-LILY/FeTaQA/blob/main/end2end/train.py
def postprocess_text(preds, references_s, metric_name):
    preds = [pred.strip() for pred in preds]
    references_s = [[reference.strip() for reference in references] for references in references_s]

    # rougeLSum expects newline after each sentence
    if metric_name == "bleurt":
        references_s = [references[0] for references in references_s]
    elif metric_name in ["sacrebleu"]:
        # since hf sacrebleu only support references with same length, we have to pad them into the same length
        ref_max_len = max([len(ref) for ref in references_s])
        for ref in references_s:
            for _ in range(ref_max_len - len(ref)):
                ref.append(None)  # see https://github.com/mjpost/sacrebleu/pull/132
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

            res = metric.compute(predictions=processed_preds, references=processed_golds)
            if metric_name == "sacrebleu":
                summary[metric_name] = res["score"] * 0.01
            elif metric_name == "bleurt":
                summary["bleurt"] = round(1.0 * sum(res["scores"]) / len(res["scores"]), 2)
            else:
                summary[metric_name] = res[metric_name]
        return summary
