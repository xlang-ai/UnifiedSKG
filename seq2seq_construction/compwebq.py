import os
import torch
from copy import deepcopy
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
        train_dataset = TrainDataset(self.args, raw_datasets['train'], cache_root)
        dev_dataset = DevDataset(self.args, raw_datasets['validation'], cache_root)
        test_dataset = TestDataset(self.args, raw_datasets['test'], cache_root)

        return train_dataset, dev_dataset, test_dataset


"""
{
    "id": datasets.Value("string"),
    "question": datasets.Value("string"),
    "answers": datasets.features.Sequence(datasets.Value("string")),
    "kg_tuples": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string"))),
}
"""


def serialize_kg_tuples(kg_tuples: list) -> str:
    # [[head, rel, tail], [head, rel, tail]]  ->  "head rel tail | head rel tail"
    return " | ".join([" ".join(t) for t in kg_tuples])


def kgqa_get_input(question: str, kg_tuples: list) -> str:
    serialized_kg = serialize_kg_tuples(kg_tuples)
    return question.strip(), serialized_kg.strip()


class TrainDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'compwebq_train.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.data = torch.load(cache_path)
        else:
            self.data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = deepcopy(raw_data)
                question = extend_data["question"]
                answers = extend_data["answers"]
                kg_tuples = extend_data["kg_tuples"]
                question, serialized_kg = kgqa_get_input(question, kg_tuples)
                # seq_out = extend_data["sparql"]
                seq_out = ', '.join(answers)

                extend_data.update({"struct_in": serialized_kg, "text_in": question, "seq_out": seq_out})
                self.data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DevDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'compwebq_dev.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.data = torch.load(cache_path)
        else:
            self.data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = deepcopy(raw_data)
                question = extend_data["question"]
                answers = extend_data["answers"]
                kg_tuples = extend_data["kg_tuples"]
                question, serialized_kg = kgqa_get_input(question, kg_tuples)
                # seq_out = extend_data["sparql"]
                seq_out = ', '.join(answers)

                extend_data.update({"struct_in": serialized_kg, "text_in": question, "seq_out": seq_out})
                self.data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def __len__(self):
        return len(self.data)


class TestDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'compwebq_test.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.data = torch.load(cache_path)
        else:
            self.data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = deepcopy(raw_data)
                question = extend_data["question"]
                answers = extend_data["answers"]
                kg_tuples = extend_data["kg_tuples"]
                question, serialized_kg = kgqa_get_input(question, kg_tuples)
                # seq_out = extend_data["sparql"]
                seq_out = ', '.join(answers)

                extend_data.update({"struct_in": serialized_kg, "text_in": question, "seq_out": seq_out})
                self.data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def __len__(self):
        return len(self.data)
