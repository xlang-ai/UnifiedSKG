import copy
import os

import torch
from datasets import DatasetDict
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from tqdm import tqdm


class Constructor(object):
    def __init__(self, args):
        self.args = args

    def to_seq2seq(self, raw_datasets: DatasetDict, cache_root: str):
        if not len(raw_datasets) == 3:
            raise AssertionError("Train, Dev, Test sections of dataset expected.")
        train_dataset = TrainDataset(self.args, raw_datasets["train"], cache_root)
        dev_dataset = DevDataset(self.args, raw_datasets["validation"], cache_root)
        test_dataset = TestDataset(self.args, raw_datasets["test"], cache_root)

        return train_dataset, dev_dataset, test_dataset


"""
datasets.Features(
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
        "expanded_turn_belief": datasets.Sequence(
                                {
                                    "slot": datasets.Value("string"),
                                    "value": datasets.Value("string")
                                })
    }
)
"""


def get_constructed_history_and_golden_response(usr_utterances, sys_utterances):
    """
    This function construct the reversed order concat of dialogue history from dialogues from users and system.
    as well as the last response(gold response) from user.
    @param usr_utterances:
    @param sys_utterances:
    @return:
    """
    # "[prefix] [utterance n] || [sys_utterance n-1] [utterance n-1] | [sys_utterance n-2] [usr_utterance n-2] | ..."
    assert len(usr_utterances) == len(sys_utterances)

    reversed_utterance_head = [
        usr_utt.strip() + " | " + sys_utt.strip()
        for usr_utt, sys_utt in zip(
            reversed(usr_utterances[:-1]), reversed(sys_utterances[:-1])
        )
    ]

    reversed_utterance_head_str = usr_utterances[-1] + " || " + sys_utterances[-1] + " | " + " | ".join(
        reversed_utterance_head)
    return reversed_utterance_head_str


class TrainDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, "multiwoz_train.cache")
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.extended_data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = copy.deepcopy(raw_data)

                history = get_constructed_history_and_golden_response(extend_data['dialog']['usr'],
                                                                      extend_data['dialog']['sys'])
                slot_ontology_values_str = ""
                for ontology_slot, ontology_values in zip(extend_data["ontology_slots"],
                                                          extend_data["ontology_values"]):
                    if not ontology_values:
                        ontology_item = "{}: {}".format(ontology_slot, "none")
                    else:
                        ontology_item = "{}: {}".format(ontology_slot, ", ".join(ontology_values))
                    slot_ontology_values_str += "{}; ".format(ontology_item)

                if (not args.seq2seq.mode) or (args.seq2seq.mode == "sequential"):
                    output_text = ", ".join(["{}-{}".format(slot, value).replace("-", " ") for slot, value in
                                             zip(extend_data['expanded_turn_belief']['slot'],
                                                 extend_data['expanded_turn_belief']['value'])])

                    extend_data.update(
                        {
                            "struct_in": slot_ontology_values_str.lower(),
                            "text_in": history.lower(),
                            "seq_out": output_text.lower(),
                        }
                    )
                    self.extended_data.append(extend_data)

                elif args.seq2seq.mode == "separate":
                    for slot, value in zip(extend_data['expanded_turn_belief']['slot'],
                                           extend_data['expanded_turn_belief']['value']):
                        # When changing the order of "sk input, question and context", we need to modify here too.
                        # we admit it was our mistake of design in that part.
                        slot_history = "{}: {}".format(slot, history)
                        output_text = value

                        extend_extend_data = copy.deepcopy(extend_data)
                        del extend_extend_data['expanded_turn_belief']
                        del extend_extend_data['ontology_slots']
                        del extend_extend_data['ontology_values']

                        extend_extend_data.update(
                            {
                                "struct_in": slot_ontology_values_str.lower(),
                                "text_in": slot_history.lower(),
                                "seq_out": output_text.lower(),
                                "slot": slot
                            }
                        )
                        self.extended_data.append(extend_extend_data)

                else:
                    raise ValueError("Other seq2seq method not support yet!")
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)


class DevDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, "multiwoz_dev.cache")
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.extended_data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = copy.deepcopy(raw_data)

                history = get_constructed_history_and_golden_response(extend_data['dialog']['usr'],
                                                                      extend_data['dialog']['sys'])
                slot_ontology_values_str = ""
                for ontology_slot, ontology_values in zip(extend_data["ontology_slots"],
                                                          extend_data["ontology_values"]):
                    if not ontology_values:
                        ontology_item = "{}: {}".format(ontology_slot, "none")
                    else:
                        ontology_item = "{}: {}".format(ontology_slot, ", ".join(ontology_values))
                    slot_ontology_values_str += "{}; ".format(ontology_item)

                if (not args.seq2seq.mode) or (args.seq2seq.mode == "sequential"):
                    output_text = ", ".join(["{}-{}".format(slot, value).replace("-", " ") for slot, value in
                                             zip(extend_data['expanded_turn_belief']['slot'],
                                                 extend_data['expanded_turn_belief']['value'])])

                    extend_data.update(
                        {
                            "struct_in": slot_ontology_values_str.lower(),
                            "text_in": history.lower(),
                            "seq_out": output_text.lower(),
                        }
                    )
                    self.extended_data.append(extend_data)

                elif args.seq2seq.mode == "separate":
                    for slot, value in zip(extend_data['expanded_turn_belief']['slot'],
                                           extend_data['expanded_turn_belief']['value']):
                        # When changing the order of "sk input, question and context", we need to modify here too.
                        # we admit it was our mistake of design in that part.
                        slot_history = "{}: {}".format(slot, history)
                        output_text = value

                        extend_extend_data = copy.deepcopy(extend_data)
                        del extend_extend_data['expanded_turn_belief']
                        del extend_extend_data['ontology_slots']
                        del extend_extend_data['ontology_values']

                        extend_extend_data.update(
                            {
                                "struct_in": slot_ontology_values_str.lower(),
                                "text_in": slot_history.lower(),
                                "seq_out": output_text.lower(),
                                "slot": slot
                            }
                        )

                        self.extended_data.append(extend_extend_data)

                else:
                    raise ValueError("Other seq2seq method not support yet!")

            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)


class TestDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, "multiwoz_test.cache")
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.extended_data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = copy.deepcopy(raw_data)

                history = get_constructed_history_and_golden_response(extend_data['dialog']['usr'],
                                                                      extend_data['dialog']['sys'])
                slot_ontology_values_str = ""
                for ontology_slot, ontology_values in zip(extend_data["ontology_slots"],
                                                          extend_data["ontology_values"]):
                    if not ontology_values:
                        ontology_item = "{}: {}".format(ontology_slot, "none")
                    else:
                        ontology_item = "{}: {}".format(ontology_slot, ", ".join(ontology_values))
                    slot_ontology_values_str += "{}; ".format(ontology_item)

                if (not args.seq2seq.mode) or (args.seq2seq.mode == "sequential"):
                    output_text = ", ".join(["{}-{}".format(slot, value).replace("-", " ") for slot, value in
                                             zip(extend_data['expanded_turn_belief']['slot'],
                                                 extend_data['expanded_turn_belief']['value'])])

                    extend_data.update(
                        {
                            "struct_in": slot_ontology_values_str.lower(),
                            "text_in": history.lower(),
                            "seq_out": output_text.lower(),
                        }
                    )
                    self.extended_data.append(extend_data)

                elif args.seq2seq.mode == "separate":
                    for slot, value in zip(extend_data['expanded_turn_belief']['slot'],
                                           extend_data['expanded_turn_belief']['value']):
                        # When changing the order of "sk input, question and context", we need to modify here too.
                        # we admit it was our mistake of design in that part.
                        slot_history = "{}: {}".format(slot, history)
                        output_text = value

                        extend_extend_data = copy.deepcopy(extend_data)
                        del extend_extend_data['expanded_turn_belief']
                        del extend_extend_data['ontology_slots']
                        del extend_extend_data['ontology_values']

                        extend_extend_data.update(
                            {
                                "struct_in": slot_ontology_values_str.lower(),
                                "text_in": slot_history.lower(),
                                "seq_out": output_text.lower(),
                                "slot": slot
                            }
                        )
                        self.extended_data.append(extend_extend_data)

                else:
                    raise ValueError("Other seq2seq method not support yet!")

            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)
