import copy
import os

import torch
from datasets import DatasetDict
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import AutoTokenizer
from tqdm import tqdm
from utils.processor import get_default_processor


class Constructor(object):
    def __init__(self, args):
        self.args = args

    def to_seq2seq(self, raw_datasets: DatasetDict, cache_root: str):
        if not len(raw_datasets) == 3:
            raise AssertionError("Train, Dev, Test sections of dataset expected.")
        train_dataset = TrainDataset(self.args, raw_datasets['train'], cache_root)
        dev_dataset = DevDataset(self.args, raw_datasets['validation'], cache_root)
        test_dataset = TestDataset(self.args, raw_datasets['test'], cache_root)

        return train_dataset, dev_dataset, test_dataset


"""
Raw data are formatted as:
{
    "id": datasets.Value("int32"),
    "dialogue":
        datasets.Sequence({
            "driver": datasets.Value("string"),
            "assistant": datasets.Value("string")
        }),
    "kb": {
        "header": datasets.Sequence(datasets.Value("string")),
        "rows": datasets.Sequence((datasets.Sequence(datasets.Value("string"))))
    },
    "intent": datasets.Value("string"),
    "global_entities": datasets.Sequence(datasets.Value("string"))
}
"""


# TODO: need to expand the history.
def kvret_get_constructed_history_and_golden_response(usr_utterances, sys_utterances):
    """
    This function construct the reversed order concat of dialogue history from dialogues from users and system.
    as well as the last response(gold response) from user.
    @param usr_utterances:
    @param sys_utterances:
    @return:
    """
    # "[prefix] [utterance n] || [sys_utterance n-1] [utterance n-1] | [sys_utterance n-2] [usr_utterance n-2] | ..."
    assert len(usr_utterances) == len(sys_utterances)

    reversed_utterance_head = [sys_utt.strip() + " | " + usr_utt.strip() for sys_utt, usr_utt in
                               zip(reversed(sys_utterances[:-1]), reversed(usr_utterances[:-1]))]

    reversed_utterance_head_str = " | ".join(reversed_utterance_head)

    return usr_utterances[-1].strip() + " || " + reversed_utterance_head_str, sys_utterances[-1]


class TrainDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets
        self.tab_processor = get_default_processor(max_cell_length=100,
                                                   # the max_cell_length is bigger in the KVRET,
                                                   # since it have large cell in table of intent weather
                                                   tokenizer=AutoTokenizer.from_pretrained(args.bert.location, use_fast=False),
                                                   max_input_length=args.seq2seq.table_truncation_max_length)

        cache_path = os.path.join(cache_root, 'kvret_train.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.extended_data = []
            expansion = args.seq2seq.expansion if args.seq2seq.expansion else 1
            for expand_id in range(expansion):
                for raw_data in tqdm(self.raw_datasets):
                    # Expand the dialogue data
                    for i in range(1, len(raw_data['dialogue']['driver']) + 1):
                        if i > min(len(raw_data['dialogue']['driver']), len(raw_data['dialogue']['assistant'])) \
                                and not len(raw_data['dialogue']['driver']) == len(raw_data['dialogue']['assistant']):
                            continue
                        # this is because some dialogue items of kvret dataset miss the response in the last turn
                        extend_data = copy.deepcopy(raw_data)
                        extend_data['dialogue']['driver'] = extend_data['dialogue']['driver'][:i]
                        extend_data['dialogue']['assistant'] = extend_data['dialogue']['assistant'][:i]
                        history, gold_response = kvret_get_constructed_history_and_golden_response(
                            usr_utterances=extend_data['dialogue']['driver'],
                            sys_utterances=extend_data['dialogue']['assistant'])
                        table_context = {"header": extend_data["kb"]["header"], "rows": extend_data["kb"]["rows"]}
                        # modify a table internally
                        for truncate_func in self.tab_processor.table_truncate_funcs:
                            truncate_func.truncate_table(table_context, history, [])
                        # linearize a table into a string
                        linear_table = self.tab_processor.table_linearize_func.process_table(table_context)

                        extend_data.update({"struct_in": linear_table.lower(),
                                            "text_in": history.lower(),
                                            "seq_out": gold_response.lower()})
                        self.extended_data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)


class DevDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets
        self.tab_processor = get_default_processor(max_cell_length=100,
                                                   tokenizer=AutoTokenizer.from_pretrained(args.bert.location, use_fast=False),
                                                   max_input_length=args.seq2seq.table_truncation_max_length)

        cache_path = os.path.join(cache_root, 'kvret_dev.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.extended_data = []
            expansion = args.seq2seq.expansion if args.seq2seq.expansion else 1
            for expand_id in range(expansion):
                for raw_data in tqdm(self.raw_datasets):
                    # Expand the dialogue data
                    for i in range(1, len(raw_data['dialogue']['driver']) + 1):
                        if i > min(len(raw_data['dialogue']['driver']), len(raw_data['dialogue']['assistant'])) \
                                and not len(raw_data['dialogue']['driver']) == len(raw_data['dialogue']['assistant']):
                            continue
                            # this is because some dialogue items of kvret dataset miss the response in the last turn
                        extend_data = copy.deepcopy(raw_data)
                        extend_data['dialogue']['driver'] = extend_data['dialogue']['driver'][:i]
                        extend_data['dialogue']['assistant'] = extend_data['dialogue']['assistant'][:i]
                        history, gold_response = kvret_get_constructed_history_and_golden_response(
                            usr_utterances=extend_data['dialogue']['driver'],
                            sys_utterances=extend_data['dialogue']['assistant'])
                        table_context = {"header": extend_data["kb"]["header"], "rows": extend_data["kb"]["rows"]}
                        # modify a table internally
                        for truncate_func in self.tab_processor.table_truncate_funcs:
                            truncate_func.truncate_table(table_context, history, [])
                        # linearize a table into a string
                        linear_table = self.tab_processor.table_linearize_func.process_table(table_context)

                        extend_data.update({"struct_in": linear_table.lower(),
                                            "text_in": history.lower(),
                                            "seq_out": gold_response.lower()})
                        self.extended_data.append(extend_data)

            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)


class TestDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets
        self.tab_processor = get_default_processor(max_cell_length=100,
                                                   tokenizer=AutoTokenizer.from_pretrained(args.bert.location, use_fast=False),
                                                   max_input_length=args.seq2seq.table_truncation_max_length)
        cache_path = os.path.join(cache_root, 'kvret_test.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.extended_data = []
            expansion = args.seq2seq.expansion if args.seq2seq.expansion else 1
            for expand_id in range(expansion):
                for raw_data in tqdm(self.raw_datasets):
                    # Expand the dialogue data
                    for i in range(1, len(raw_data['dialogue']['driver']) + 1):
                        if i > min(len(raw_data['dialogue']['driver']), len(raw_data['dialogue']['assistant'])) \
                                and not len(raw_data['dialogue']['driver']) == len(raw_data['dialogue']['assistant']):
                            continue
                            # this is because some dialogue items of kvret dataset miss the response in the last turn
                        extend_data = copy.deepcopy(raw_data)
                        extend_data['dialogue']['driver'] = extend_data['dialogue']['driver'][:i]
                        extend_data['dialogue']['assistant'] = extend_data['dialogue']['assistant'][:i]
                        history, gold_response = kvret_get_constructed_history_and_golden_response(
                            usr_utterances=extend_data['dialogue']['driver'],
                            sys_utterances=extend_data['dialogue']['assistant'])
                        table_context = {"header": extend_data["kb"]["header"], "rows": extend_data["kb"]["rows"]}
                        # modify a table internally
                        for truncate_func in self.tab_processor.table_truncate_funcs:
                            truncate_func.truncate_table(table_context, history, [])
                        # linearize a table into a string
                        linear_table = self.tab_processor.table_linearize_func.process_table(table_context)

                        extend_data.update({"struct_in": linear_table.lower(),
                                            "text_in": history.lower(),
                                            "seq_out": gold_response.lower()})
                        self.extended_data.append(extend_data)

            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)
