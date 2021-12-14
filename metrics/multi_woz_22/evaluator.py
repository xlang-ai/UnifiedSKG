# encoding=utf8
import json
import os
from collections import OrderedDict

import nltk
from datasets import load_metric


def load_entities(multi_woz_22_entity_file_paths: list):
    """

    @param multi_woz_22_entity_file_paths: a list of .json which we can load kb/entities
    @return:
    """
    under_scored_entity_dict = OrderedDict()
    for multi_woz_22_entity_file_path in multi_woz_22_entity_file_paths:
        with open(multi_woz_22_entity_file_path) as f:
            # FIXME: ask dialogue expert for whether this mean of entities extraction is right
            entities = json.load(f)
            for entity_item in entities:
                for entity_name, entity_value in entity_item.items():
                    if isinstance(entity_value, str):
                        under_scored_entity_dict[entity_value] = entity_value.replace(" ", "_")

    return under_scored_entity_dict


def postprocess_text(preds, responses, metric_name):
    preds = [pred.strip() for pred in preds]
    responses = [response.strip() for response in responses]

    # rougeLSum expects newline after each sentence
    if metric_name == "rouge":
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        responses = ["\n".join(nltk.sent_tokenize(response)) for response in responses]
    elif metric_name == "sacrebleu":  # sacrebleu
        responses = [[response] for response in responses]
    elif metric_name == "bleu":
        preds = [pred.split(" ") for pred in preds]
        responses = [[response.split(" ")] for response in responses]
    else:
        pass

    return preds, responses


class EvaluateTool(object):
    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):
        summary = {}
        global_entities = load_entities(
            [path for path in os.listdir(golds[0]["db_root_path"]) if str(path).endswith(".json")])

        gold_responses = [
            {
                "response": item["seq_out"],
                "intents": item["services"],
            }
            for item in golds
        ]
        assert len(preds) == len(gold_responses)

        metric_list = []
        if section in ["train", "dev"]:
            metric_list = ["bleu"]
        elif section == "test":
            metric_list = ["bleu", "metrics/added/response_entity_hit.py"]

        for metric_name in metric_list:
            metric = load_metric(metric_name)
            processed_preds, processed_golds = postprocess_text(
                preds, gold_responses, metric_name
            )
            res = metric.compute(
                predictions=processed_preds,
                references=processed_golds,
                global_entities=global_entities,
            )
            if metric_name == "metrics/added/response_entity_hit.py":
                summary.update(res)
            else:
                summary[metric_name] = res[metric_name]

        return summary
