# encoding=utf8
import json

from .multiwoz_metrics import evaluate_metrics, get_slot_information

"""
{
    "ID": datasets.Value("string"),
    "turn_id": datasets.Value("int32"),
    "ontology_path": datasets.Value("string"),
    "dialog": {
        "sys": datasets.Sequence(datasets.Value("string")),
        "usr": datasets.Sequence(datasets.Value("string"))
    },
    "domains": datasets.Sequence(datasets.Value("string")),
    "ontology_slots": datasets.Sequence(datasets.Value("string")),
    "ontology_values": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
    "turn_belief": datasets.Sequence(datasets.Value("string")),
    "expanded_turn_belief": datasets.Sequence(datasets.Value("string"))
}
"""


def postprocess_text(mode, preds, golds):
    predictions = {}
    for pred, gold in zip(preds, golds):
        dial_id = gold["ID"]
        if dial_id not in predictions:
            predictions[dial_id] = {}
            predictions[dial_id]["domains"] = gold["domains"]
            predictions[dial_id]["turns"] = {}

        cleaned_gold_belief = []
        for bs in gold["turn_belief"]:
            if 'not mentioned' in bs or 'none' in bs:
                continue
            cleaned_gold_belief.append(bs)

        if gold["turn_id"] not in predictions[dial_id]["turns"].keys():
            predictions[dial_id]["turns"][gold["turn_id"]] = {"turn_belief": cleaned_gold_belief, "pred_belief": []}

        for pred_slot_value in pred.split(", "):
            if len(pred_slot_value.split(" ")) < 2:
                continue
            pred_tokens = pred_slot_value.split(" ")
            if pred_tokens[1] == "book":
                domain_slot_pred = pred_tokens[0] + "-" + " ".join(pred_tokens[1:3])
                value_pred = " ".join(pred_slot_value.split(" ")[3:])
            else:
                domain_slot_pred = pred_tokens[0] + "-" + pred_tokens[1]
                value_pred = " ".join(pred_slot_value.split(" ")[2:])
            pred_bs = "{}-{}".format(domain_slot_pred, value_pred)
            if 'not mentioned' in pred_bs or 'none' in pred_bs:
                continue
            predictions[dial_id]["turns"][gold["turn_id"]]["pred_belief"].append(pred_bs)

    return predictions


class EvaluateTool(object):
    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):
        summary = {}

        ontology_path = golds[0]['ontology_path']
        ontology = json.load(open(ontology_path, 'r'))
        ALL_SLOTS = get_slot_information(ontology)

        predictions = postprocess_text(self.args.seq2seq.mode, preds, golds)
        joint_acc_score, F1_score, turn_acc_score = evaluate_metrics(predictions, ALL_SLOTS)
        if section in ["train", "dev"]:
            summary.update({"Joint Acc": joint_acc_score})
        elif section == "test":
            summary.update({"Joint Acc": joint_acc_score, "Turn Acc": turn_acc_score, "Joint F1": F1_score})

        return summary
