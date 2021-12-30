import os
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from datasets.dataset_dict import DatasetDict
from tqdm import tqdm

class Constructor(object):
    def __init__(self, args):
        self.args = args

    def to_seq2seq(self, raw_datasets: DatasetDict, cache_root: str):
        if not len(raw_datasets) == 3:
            raise AssertionError("Train, Dev sections of dataset expected.")
        train_dataset = TrainDataset(self.args, raw_datasets["train"], cache_root)
        dev_dataset = DevDataset(self.args, raw_datasets["validation"], cache_root)
        test_dataset = TestDataset(self.args, raw_datasets["test"], cache_root)

        return train_dataset, dev_dataset, test_dataset


class TrainDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        self.args = args
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'sql2text_train.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.data = torch.load(cache_path)
        else:
            self.data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = deepcopy(raw_data)

                extend_data.update({"struct_in": "",
                                    "text_in": extend_data["query"],
                                    "seq_out": extend_data["question"].strip()})
                self.data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DevDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        self.args = args
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'sql2text_dev.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.data = torch.load(cache_path)
        else:
            self.data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = deepcopy(raw_data)

                extend_data.update({"struct_in": "",
                                    "text_in": extend_data["query"],
                                    "seq_out": extend_data["question"].strip()})
                self.data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def __len__(self):
        return len(self.data)


class TestDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        self.args = args
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'sql2text_test.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.data = torch.load(cache_path)
        else:
            self.data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = deepcopy(raw_data)

                extend_data.update({"struct_in": "",
                                    "text_in": extend_data["query"],
                                    "seq_out": extend_data["question"].strip()})
                self.data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def __len__(self):
        return len(self.data)
