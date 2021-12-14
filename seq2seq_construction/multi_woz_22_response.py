import copy
import json
import os

import torch
from datasets import DatasetDict
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import AutoTokenizer

from utils.processor import get_default_processor


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
        "dialogue_id": datasets.Value("string"),
        "db_root_path": datasets.Value("string"),
        "services": datasets.Sequence(datasets.Value("string")),
        "db_paths": datasets.Sequence(datasets.Value("string")),
        "turns": datasets.Sequence(
            {
                "turn_id": datasets.Value("string"),
                "speaker": datasets.ClassLabel(names=["USER", "SYSTEM"]),
                "utterance": datasets.Value("string"),
                "frames": no_use_in_our_setting,
                "dialogue_acts": no_use_in_our_setting ,
            }
        ),
    }
)
"""


def load_db(db_paths: list, return_proportion=True):
    """
    Load dbs in Multi-WoZ dialogue.
    @param db_paths: a list of db_path since each dialogue in Multi-WoZ may have multiple db
    @param return_proportion: whether to return the length proportion of each table in all db_tables
    @return: the db table format like
    """
    # load the db tables from json file
    db_tables = []

    for db_path in db_paths:
        if db_path.endswith("taxi_db.json"):
            continue  # we will skip loading the db of taxi_db since it has some issue.
        with open(os.path.join(db_path)) as f:
            comment_removed_raw_str = "".join(
                [line for line in f.readlines() if not line.startswith("#")]
            )
            # have to do so since json file like hospital_db.json have comment in the front of file.
            db = json.loads(comment_removed_raw_str)
        header = list(db[0].keys())
        rows = []
        for db_item in db:
            row = []
            for column_name, cell_value in db_item.items():
                if isinstance(cell_value, list):
                    row.append(str(tuple(cell_value)))
                elif isinstance(cell_value, dict):
                    row.append(
                        ", ".join(
                            ["{}: {}".format(k, v) for k, v in cell_value.items()]
                        )
                    )
                elif isinstance(cell_value, str):
                    row.append(cell_value)

            rows.append(row)
        db_table = {"header": header, "rows": rows}
        db_tables.append(db_table)

    proportions = []
    for db_table in db_tables:
        table_length = len(db_table["header"])
        for row in db_table["rows"]:
            table_length += len(row)
        proportions.append(table_length)
    total = sum(proportions)
    proportions = [proportion / total for proportion in proportions]
    if return_proportion:
        return db_tables, proportions
    else:
        return db_tables


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
        sys_utt.strip() + " | " + usr_utt.strip()
        for sys_utt, usr_utt in zip(
            reversed(sys_utterances[:-1]), reversed(usr_utterances[:-1])
        )
    ]

    reversed_utterance_head_str = " | ".join(reversed_utterance_head)

    return (
        usr_utterances[-1].strip() + " || " + reversed_utterance_head_str,
        usr_utterances[-1],
    )


class TrainDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        self.raw_datasets = raw_datasets
        tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=False)

        cache_path = os.path.join(cache_root, "multi_woz_22_train.cache")
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.extended_data = []
            for raw_data in self.raw_datasets:
                # Expand the dialogue data
                for i in range(len(raw_data["turns"])):
                    extend_data = copy.deepcopy(raw_data)
                    extend_data["usr"] = [
                        turn["utterance"]
                        for turn in extend_data["turn"]
                        if turn["speaker"] == "USER"
                    ]
                    extend_data["sys"] = [
                        turn["utterance"]
                        for turn in extend_data["turn"]
                        if turn["speaker"] == "SYSTEM"
                    ]
                    extend_data["usr"] = extend_data["usr"][:i]
                    extend_data["sys"] = extend_data["sys"][:i]
                    (
                        history,
                        gold_response,
                    ) = get_constructed_history_and_golden_response(
                        usr_utterances=extend_data["usr"],
                        sys_utterances=extend_data["sys"],
                    )

                    db_tables, proportions = load_db(raw_data["db_paths"])

                    linear_table_s = []

                    history_length = len(tokenizer.tokenize(history))
                    table_truncation_max_length_for_table = (
                            args.seq2seq.table_truncation_max_length
                            - history_length
                    )
                    for table_context, proportion, table_name in zip(db_tables, proportions, raw_data["services"]):
                        tab_processor = get_default_processor(
                            max_cell_length=200,
                            # the max_cell_length is bigger in the MultiWoZ,
                            # e.g. you can check "openhours" in "db/attraction_db.json"
                            tokenizer=tokenizer,
                            max_input_length=int(
                                table_truncation_max_length_for_table * proportion + history_length
                            ),
                            # MARK*: We assign the max length by proportion of each table
                        )

                        # modify a table internally
                        for truncate_func in tab_processor.table_truncate_funcs:
                            truncate_func.truncate_table(table_context, history, [])
                        # linearize a table into a string
                        linear_table = tab_processor.table_linearize_func.process_table(
                            table_context
                        )
                        linear_table = "{}: {}".format(table_name, linear_table)
                        linear_table_s.append(linear_table)

                    linear_tables = " || ".join(linear_table_s)

                    extend_data.update(
                        {
                            "struct_in": linear_tables.lower(),
                            "text_in": history.lower(),
                            "seq_out": gold_response.lower(),
                        }
                    )
                    self.extended_data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)


class DevDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        self.raw_datasets = raw_datasets
        tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=False)

        cache_path = os.path.join(cache_root, "multi_woz_22_dev.cache")
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.extended_data = []
            for raw_data in self.raw_datasets:
                # Expand the dialogue data
                for i in range(len(raw_data["turns"])):
                    extend_data = copy.deepcopy(raw_data)
                    extend_data["usr"] = [
                        turn["utterance"]
                        for turn in extend_data["turn"]
                        if turn["speaker"] == "USER"
                    ]
                    extend_data["sys"] = [
                        turn["utterance"]
                        for turn in extend_data["turn"]
                        if turn["speaker"] == "SYSTEM"
                    ]
                    extend_data["usr"] = extend_data["usr"][:i]
                    extend_data["sys"] = extend_data["sys"][:i]
                    (
                        history,
                        gold_response,
                    ) = get_constructed_history_and_golden_response(
                        usr_utterances=extend_data["usr"],
                        sys_utterances=extend_data["sys"],
                    )

                    db_tables, proportions = load_db(raw_data["db_paths"])

                    linear_table_s = []

                    history_length = len(tokenizer.tokenize(history))
                    table_truncation_max_length_for_table = (
                            args.seq2seq.table_truncation_max_length
                            - history_length
                    )
                    for table_context, proportion, table_name in zip(db_tables, proportions, raw_data["services"]):
                        tab_processor = get_default_processor(
                            max_cell_length=200,
                            # the max_cell_length is bigger in the MultiWoZ,
                            # e.g. you can check "openhours" in "db/attraction_db.json"
                            tokenizer=tokenizer,
                            max_input_length=int(
                                table_truncation_max_length_for_table * proportion + history_length
                            ),
                            # MARK*: We assign the max length by proportion of each table
                        )

                        # modify a table internally
                        for truncate_func in tab_processor.table_truncate_funcs:
                            truncate_func.truncate_table(table_context, history, [])
                        # linearize a table into a string
                        linear_table = tab_processor.table_linearize_func.process_table(
                            table_context
                        )
                        linear_table = "{}: {}".format(table_name, linear_table)
                        linear_table_s.append(linear_table)

                    linear_tables = " || ".join(linear_table_s)

                    extend_data.update(
                        {
                            "struct_in": linear_tables.lower(),
                            "text_in": history.lower(),
                            "seq_out": gold_response.lower(),
                        }
                    )
                    self.extended_data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)


class TestDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        self.raw_datasets = raw_datasets
        tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=False)

        cache_path = os.path.join(cache_root, "multi_woz_22_test.cache")
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.extended_data = []
            for raw_data in self.raw_datasets:
                # Expand the dialogue data
                for i in range(len(raw_data["turns"])):
                    extend_data = copy.deepcopy(raw_data)
                    extend_data["usr"] = [
                        turn["utterance"]
                        for turn in extend_data["turn"]
                        if turn["speaker"] == "USER"
                    ]
                    extend_data["sys"] = [
                        turn["utterance"]
                        for turn in extend_data["turn"]
                        if turn["speaker"] == "SYSTEM"
                    ]
                    extend_data["usr"] = extend_data["usr"][:i]
                    extend_data["sys"] = extend_data["sys"][:i]
                    (
                        history,
                        gold_response,
                    ) = get_constructed_history_and_golden_response(
                        usr_utterances=extend_data["usr"],
                        sys_utterances=extend_data["sys"],
                    )

                    db_tables, proportions = load_db(raw_data["db_paths"])

                    linear_table_s = []

                    history_length = len(tokenizer.tokenize(history))
                    table_truncation_max_length_for_table = (
                            args.seq2seq.table_truncation_max_length
                            - history_length
                    )
                    for table_context, proportion, table_name in zip(db_tables, proportions, raw_data["services"]):
                        tab_processor = get_default_processor(
                            max_cell_length=200,
                            # the max_cell_length is bigger in the MultiWoZ,
                            # e.g. you can check "openhours" in "db/attraction_db.json"
                            tokenizer=tokenizer,
                            max_input_length=int(
                                table_truncation_max_length_for_table * proportion + history_length
                            ),
                            # MARK*: We assign the max length by proportion of each table
                        )

                        # modify a table internally
                        for truncate_func in tab_processor.table_truncate_funcs:
                            truncate_func.truncate_table(table_context, history, [])
                        # linearize a table into a string
                        linear_table = tab_processor.table_linearize_func.process_table(
                            table_context
                        )
                        linear_table = "{}: {}".format(table_name, linear_table)
                        linear_table_s.append(linear_table)

                    linear_tables = " || ".join(linear_table_s)

                    extend_data.update(
                        {
                            "struct_in": linear_tables.lower(),
                            "text_in": history.lower(),
                            "seq_out": gold_response.lower(),
                        }
                    )
                    self.extended_data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)
