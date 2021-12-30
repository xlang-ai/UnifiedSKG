# encoding=utf8
from collections import OrderedDict
import json
import nltk
from datasets import load_metric


def load_entities(kvret_entity_file_path):
    """

    @param kvret_entity_file_path: the path of kvret_entities.json
    @return:
    """
    under_scored_entity_dict = OrderedDict()
    with open(kvret_entity_file_path) as f:
        entity = json.load(f)
        for sub_class_name, sub_class_entity_list in entity.items():
            if sub_class_name == 'poi':
                for entity_item in sub_class_entity_list:
                    under_scored_entity_dict[str(entity_item['address'])] = (
                        str(entity_item['address']).replace(" ", "_"))
                    under_scored_entity_dict[str(entity_item['poi'])] = (str(entity_item['poi']).replace(" ", "_"))
                    under_scored_entity_dict[str(entity_item['type'])] = (str(entity_item['type']).replace(" ", "_"))
            elif sub_class_name == "distance":
                for entity_item in sub_class_entity_list:
                    under_scored_entity_dict[str(entity_item) + " miles"] = str(entity_item) + " miles"
            elif sub_class_name == "temperature":
                for entity_item in sub_class_entity_list:
                    under_scored_entity_dict[str(entity_item) + "f"] = str(entity_item) + "f"
            else:
                for entity_item in sub_class_entity_list:
                    under_scored_entity_dict[str(entity_item)] = (str(entity_item).replace(" ", "_"))
        # add missing entities,
        missed_entities = ["yoga", "tennis", "swimming", "football", " lab ", "doctor", "optometrist", "dentist", "1st",
                           "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th",
                           "11th", "12th", "13th", "14th", "15th", "16th", "17th", "18th", "19th", "20th", "Jill",
                           "Jack"]
        for missed_entity in missed_entities:
            under_scored_entity_dict[str(missed_entity)] = (missed_entity)

        # special handle of "HR"
        del under_scored_entity_dict['HR']
        under_scored_entity_dict[' HR '] = ' HR '

    return under_scored_entity_dict


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

        assert len(golds) > 0
        global_entities = load_entities(golds[0]["entities_file"])

        metric_list = []
        if section in ["train", "dev"]:
            metric_list = ["bleu"]
        elif section == "test":
            metric_list = ["bleu", "metrics/kvret/response_entity_hit.py"]

        for metric_name in metric_list:
            metric = load_metric(metric_name)

            if metric_name == "metrics/kvret/response_entity_hit.py":
                gold_responses = [
                    {
                        "response": item["seq_out"],
                        "intents": [item["intent"]],
                    }
                    for item in golds
                ]
                res = metric.compute(
                    **{
                        "predictions": preds,
                        "references": gold_responses,
                        "global_entities": global_entities,
                    }
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