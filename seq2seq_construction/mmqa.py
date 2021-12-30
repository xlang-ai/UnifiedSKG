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
        if not len(raw_datasets) == 2:
            raise AssertionError("Train, Dev sections of dataset expected.")
        train_dataset = TrainDataset(self.args, raw_datasets['train'], cache_root)
        dev_dataset = DevDataset(self.args, raw_datasets['validation'], cache_root)
        # test_dataset = TestDataset(self.args, raw_datasets['test'], cache_root)

        return train_dataset, dev_dataset


"""
Data item is formatted as:
Raw data are formatted as:
  {
      "id": datasets.Value("string"),
      "question": datasets.Value("string"),
      "table_id": datasets.features.Sequence(datasets.Value("string")),
      "table": datasets.features.Sequence({"header": datasets.features.Sequence(datasets.Value("string")),
                "rows": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string")))}),
      "meta": datasets.features.Sequence(datasets.Value("string")),
      "answer_text": datasets.Value("string"),
  }
"""


class TrainDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'mmqa_train.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.tab_processor = get_default_processor(max_cell_length=15,
                                                       tokenizer=AutoTokenizer.from_pretrained(args.bert.location, use_fast=False),
                                                       max_input_length=args.seq2seq.table_truncation_max_length // 2)

            self.extended_data = []
            expansion = args.seq2seq.expansion if args.seq2seq.expansion else 1
            for expand_id in range(expansion):
                for raw_data in tqdm(self.raw_datasets):
                    extend_data = copy.deepcopy(raw_data)
                    question = extend_data["question"].lower()
                    tables = [{"header": header, "rows": rows} for header, rows
                              in zip(extend_data["table"]["header"], extend_data["table"]["rows"])]
                    contexts = extend_data["context"]
                    gold_result = extend_data["answer_text"]

                    table_contexts = copy.deepcopy(tables)
                    table_contexts_strs = []
                    for table_context in table_contexts:
                        for truncate_func in self.tab_processor.table_truncate_funcs:
                            try:
                                truncate_func.truncate_table(table_context, question, [])
                            except:
                                raise NotImplementedError()
                            table_contexts_strs.append(
                                self.tab_processor.table_linearize_func.process_table(table_context))

                    linear_table = ""
                    for meta, table_context_str in zip(extend_data["meta"], table_contexts_strs):
                        linear_table += meta + " " + table_context_str

                    passage_context_str = " ".join(contexts)
                    extend_data.update({"struct_in": linear_table.lower() + passage_context_str.lower(),
                                        "text_in": question.lower(),
                                        "seq_out": gold_result.lower()})
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

        cache_path = os.path.join(cache_root, 'mmqa_dev.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.tab_processor = get_default_processor(max_cell_length=15,
                                                       tokenizer=AutoTokenizer.from_pretrained(args.bert.location, use_fast=False),
                                                       max_input_length=args.seq2seq.table_truncation_max_length // 2)

            self.extended_data = []
            expansion = args.seq2seq.expansion if args.seq2seq.expansion else 1
            for expand_id in range(expansion):
                for raw_data in self.raw_datasets:
                    extend_data = copy.deepcopy(raw_data)
                    question = extend_data["question"].lower()
                    tables = [{"header": header, "rows": rows} for header, rows
                              in zip(extend_data["table"]["header"], extend_data["table"]["rows"])]
                    gold_result = extend_data["answer_text"]
                    contexts = extend_data["context"]

                    table_contexts = copy.deepcopy(tables)
                    table_contexts_strs = []
                    for table_context in table_contexts:
                        for truncate_func in self.tab_processor.table_truncate_funcs:
                            truncate_func.truncate_table(table_context, question, [])
                            table_contexts_strs.append(
                                self.tab_processor.table_linearize_func.process_table(table_context))

                    linear_table = ""
                    for meta, table_context_str in zip(extend_data["meta"], table_contexts_strs):
                        linear_table += meta + " " + table_context_str
                    passage_context_str = " ".join(contexts)
                    extend_data.update({"struct_in": linear_table.lower() + passage_context_str.lower(),
                                        "text_in": question.lower(),
                                        "seq_out": gold_result.lower()})
                    self.extended_data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)
