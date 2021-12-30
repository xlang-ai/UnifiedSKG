import copy
import os

import torch
from datasets import DatasetDict
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import AutoTokenizer

from utils.processor import get_default_processor
from tqdm import tqdm

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

The data is formatted as:
    {
        "id": datasets.Value("string"),
        "annotator": datasets.Value("int32"),
        "position": datasets.Value("int32"),
        "question": datasets.Value("string"),
        "table_file": datasets.Value("string"),
        "table_header": datasets.features.Sequence(datasets.Value("string")),
        "table_data": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string"))),
        "answer_coordinates": datasets.features.Sequence(
            {"row_index": datasets.Value("int32"), "column_index": datasets.Value("int32")}
        ),
        "answer_text": datasets.features.Sequence(datasets.Value("string")),
    }
"""


def sqa_get_constructed_history_and_golden_response(question_and_history):
    """"""
    reversed_utterance_head = [question.strip() for question in reversed(question_and_history[:-1])]
    reversed_utterance_head_str = " | ".join(reversed_utterance_head)
    return question_and_history[-1].strip() + " || " + reversed_utterance_head_str, question_and_history[-1]


class TrainDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'sqa_train.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.tab_processor = get_default_processor(max_cell_length=15,
                                                       tokenizer=AutoTokenizer.from_pretrained(args.bert.location, use_fast=False),
                                                       max_input_length=args.seq2seq.table_truncation_max_length)

            self.extended_data = []
            for i, raw_data in tqdm(enumerate(self.raw_datasets)):
                extended_data = copy.deepcopy(raw_data)
                question_and_history_str, question_in_this_turn = sqa_get_constructed_history_and_golden_response(
                    extended_data['question_and_history'])
                table_content = {"header": extended_data['table_header'], "rows": extended_data['table_data']}
                answer = extended_data['answer_text']

                try:
                    # modify a table internally
                    for truncate_func in self.tab_processor.table_truncate_funcs:
                        truncate_func.truncate_table(table_content, question_and_history_str, answer)
                    # linearize a table into a string
                    linear_table = self.tab_processor.table_linearize_func.process_table(table_content)
                    seq_out = self.tab_processor.process_output(answer)

                    extended_data.update({"struct_in": linear_table.lower(),
                                          "text_in": question_and_history_str.lower(),
                                          "seq_out": seq_out.lower()})
                    self.extended_data.append(extended_data)
                except:
                    print("sqa data damaged in this line.")
                    continue
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

        cache_path = os.path.join(cache_root, 'sqa_dev.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.tab_processor = get_default_processor(max_cell_length=15,
                                                       tokenizer=AutoTokenizer.from_pretrained(args.bert.location, use_fast=False),
                                                       max_input_length=args.seq2seq.table_truncation_max_length)

            self.extended_data = []
            for i, raw_data in tqdm(enumerate(self.raw_datasets)):
                extended_data = copy.deepcopy(raw_data)
                question_and_history_str, question_in_this_turn = sqa_get_constructed_history_and_golden_response(
                    extended_data['question_and_history'])
                table_content = {"header": extended_data['table_header'], "rows": extended_data['table_data']}
                answer = extended_data['answer_text']

                try:
                    # modify a table internally
                    for truncate_func in self.tab_processor.table_truncate_funcs:
                        truncate_func.truncate_table(table_content, question_and_history_str, [])
                    # linearize a table into a string
                    linear_table = self.tab_processor.table_linearize_func.process_table(table_content)
                    seq_out = self.tab_processor.process_output(answer)

                    extended_data.update({"struct_in": linear_table.lower(),
                                          "text_in": question_and_history_str.lower(),
                                          "seq_out": seq_out.lower()})
                    self.extended_data.append(extended_data)
                except:
                    print("sqa data damaged in this line.")
                    continue
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

        cache_path = os.path.join(cache_root, 'sqa_test.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.tab_processor = get_default_processor(max_cell_length=15,
                                                       tokenizer=AutoTokenizer.from_pretrained(args.bert.location, use_fast=False),
                                                       max_input_length=args.seq2seq.table_truncation_max_length)

            self.extended_data = []
            for i, raw_data in tqdm(enumerate(self.raw_datasets)):
                extended_data = copy.deepcopy(raw_data)
                question_and_history_str, question_in_this_turn = sqa_get_constructed_history_and_golden_response(
                    extended_data['question_and_history'])
                table_content = {"header": extended_data['table_header'], "rows": extended_data['table_data']}
                answer = extended_data['answer_text']

                try:
                    # modify a table internally
                    for truncate_func in self.tab_processor.table_truncate_funcs:
                        truncate_func.truncate_table(table_content, question_and_history_str, [])
                    # linearize a table into a string
                    linear_table = self.tab_processor.table_linearize_func.process_table(table_content)
                    seq_out = self.tab_processor.process_output(answer)

                    extended_data.update({"struct_in": linear_table.lower(),
                                          "text_in": question_and_history_str.lower(),
                                          "seq_out": seq_out.lower()})
                    self.extended_data.append(extended_data)
                except:
                    print("sqa data damaged in this line.")
                    continue
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)
