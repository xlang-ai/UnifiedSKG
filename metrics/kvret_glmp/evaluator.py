# encoding=utf8
from collections import OrderedDict
import json
import nltk
from datasets import load_metric

def postprocess_text(preds, responses, metric_name):
    _preds = [pred.strip() for pred in preds]
    _responses = [response.strip() for response in responses]

    # rougeLSum expects newline after each sentence
    if metric_name == "rouge":
        _preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in _preds]
        _responses = ["\n".join(nltk.sent_tokenize(response)) for response in _responses]
    elif metric_name == "sacrebleu":  # sacrebleu
        _responses = [[response] for response in _responses]
    elif metric_name == "bleu":
        _preds = [pred.split(" ") for pred in _preds]
        _responses = [[response.split(" ")] for response in _responses]
    else:
        pass

    return _preds, _responses


class EvaluateTool(object):
    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):
        summary = {}
        # Here is the post-process code from huggingface transformers tokenizer decodes,
        # it clean up clean up the space from tokenization(" 's"->"'s" e.t.c)
        # we couldn't
        # @staticmethod
        # def clean_up_tokenization(out_string: str) -> str:
        # """
        # Clean up a list of simple English tokenization artifacts like spaces
        # before punctuations and abbreviated forms.
        #
        # Args:
        #     out_string (:obj:`str`): The text to clean up.
        #
        # Returns:
        #     :obj:`str`: The cleaned-up string.
        # """
        # out_string = (
        #     out_string.replace(" .", ".")
        #     .replace(" ?", "?")
        #     .replace(" !", "!")
        #     .replace(" ,", ",")
        #     .replace(" ' ", "'")
        #     .replace(" n't", "n't")
        #     .replace(" 'm", "'m")
        #     .replace(" 's", "'s")
        #     .replace(" 've", "'ve")
        #     .replace(" 're", "'re")
        # )
        # return out_string

        # Therefore, to make fair comparison, we "convert that back".
        
        processed_preds = []
        for pred in preds:
            pred = pred + " " # for matching ". " in the end of a sentence.
            processed_preds.append(pred.replace(". ", " . ")
                                   .replace("? ", " ? ")
                                   .replace("! ", " ! ")
                                   .replace(", ", " , ")
                                   .replace("'", " ' ")
                                   .replace("n't ", " n't ")
                                   .replace("'m ", " 'm ")
                                   .replace("'s ", " 's ")
                                   .replace("'ve ", " 've ")
                                   .replace("'re ", " 're ")
                                   .strip())
        preds = processed_preds

        metric_list = []
        if section in ["train", "dev"]:
            metric_list = ["bleu"]
        elif section == "test":
            metric_list = ["bleu", "metrics/kvret_glmp/entity_metric.py"]

        for metric_name in metric_list:
            metric = load_metric(metric_name)

            if metric_name == "metrics/kvret_glmp/entity_metric.py":

                golds_sub_attrs = [{"kb_arr": gold["kb_arr"],
                                    'response': gold['response'],
                                    'ent_index': gold["ent_index"],
                                    "ent_idx_cal": gold["ent_idx_cal"],
                                    "ent_idx_wet": gold["ent_idx_wet"],
                                    "ent_idx_nav": gold["ent_idx_nav"],
                                    'domain': gold["domain"],
                                    "entities_file": gold["entities_file"]}
                                   for gold in golds]

                res = metric.compute(
                    predictions=preds,
                    references=golds_sub_attrs,
                )
                summary.update(res)
            else:
                gold_responses = [item["seq_out"] for item in golds]
                processed_preds, processed_golds = postprocess_text(
                    preds, gold_responses, metric_name
                )
                res = metric.compute(
                    predictions=processed_preds,
                    references=processed_golds,
                )
                summary[metric_name] = res[metric_name]

        return summary


if __name__ == '__main__':
    import json

    with open("predictions_large.json") as f:
        test_data = json.load(f)
    preds = [item['prediction'] for item in test_data]
    evaluator = EvaluateTool(args=None)
    score = evaluator.evaluate(preds, test_data, section="test")
    print(score)