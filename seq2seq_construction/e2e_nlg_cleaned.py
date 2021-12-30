import re
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
        train_dataset = TrainDataset(self.args, raw_datasets['train'], cache_root)
        dev_dataset = DevDataset(self.args, raw_datasets['validation'], cache_root)
        test_dataset = TestDataset(self.args, raw_datasets['test'], cache_root)

        return train_dataset, dev_dataset, test_dataset


"""
Data item is formatted as:
{
    "meaning_representation": datasets.Value("string"),
    "human_reference": datasets.Value("string"),
}
"""


def parse_meaning_representation(meaning_representation_str):
    result = {}
    mr_str_s = meaning_representation_str.split(', ')
    for mr_str in mr_str_s:
        mrc_format = "(.*)\[(.*)\]"
        attr, value = re.compile(mrc_format).match(mr_str).groups()
        result[attr] = value
    return result


class TrainDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        self.args = args
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'e2e_nlg_cleaned_train.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.full_src_lst, self.full_tgt_lst = torch.load(cache_path)
        else:
            self.full_src_lst = []
            self.full_tgt_lst = []
            for example in tqdm(self.raw_datasets):
                mr_attr_value_pairs = parse_meaning_representation(example['meaning_representation'])
                human_reference = example['human_reference']
                mr_str = ''
                for attr, value in mr_attr_value_pairs.items():
                    mr_str += "{} : {} | ".format(attr, value)

                self.full_src_lst.append(mr_str)
                self.full_tgt_lst.append(human_reference)
            if args.dataset.use_cache:
                torch.save((self.full_src_lst, self.full_tgt_lst), cache_path)

    def __getitem__(self, index) -> T_co:
        raw_data = self.raw_datasets[index]
        raw_data.update({"struct_in": self.full_src_lst[index],
                         "text_in": "",
                         "seq_out": self.full_tgt_lst[index]})
        return raw_data

    def __len__(self):
        return len(self.full_src_lst)


class DevDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        self.args = args
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'e2e_nlg_cleaned_dev.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.full_src_lst, self.full_tgt_lst = torch.load(cache_path)
        else:
            self.full_src_tgt_dic = {}

            for example in tqdm(self.raw_datasets):
                mr_attr_value_pairs = parse_meaning_representation(example['meaning_representation'])
                human_reference = example['human_reference']
                mr_str = ''
                for attr, value in mr_attr_value_pairs.items():
                    mr_str += "{} : {} | ".format(attr, value)

                if mr_str in self.full_src_tgt_dic.keys():
                    self.full_src_tgt_dic[mr_str].append(human_reference)
                else:
                    self.full_src_tgt_dic[mr_str] = [human_reference]

            self.full_src_lst = []
            self.full_tgt_lst = []
            for src, tgt in self.full_src_tgt_dic.items():
                self.full_src_lst.append(src)
                self.full_tgt_lst.append(tgt)
            if args.dataset.use_cache:
                torch.save((self.full_src_lst, self.full_tgt_lst), cache_path)

    def __getitem__(self, index) -> T_co:
        raw_data = self.raw_datasets[index]
        raw_data.update({"struct_in": self.full_src_lst[index],
                         "text_in": "",
                         "seq_out": self.full_tgt_lst[index][0],
                         "references": self.full_tgt_lst[index]})
        return raw_data

    def __len__(self):
        return len(self.full_src_lst)


class TestDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        self.args = args
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'e2e_nlg_cleaned_test.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.full_src_lst, self.full_tgt_lst = torch.load(cache_path)
        else:
            self.full_src_tgt_dic = {}

            for example in tqdm(self.raw_datasets):
                mr_attr_value_pairs = parse_meaning_representation(example['meaning_representation'])
                human_reference = example['human_reference']
                mr_str = ''
                for attr, value in mr_attr_value_pairs.items():
                    mr_str += "{} : {} | ".format(attr, value)

                if mr_str in self.full_src_tgt_dic.keys():
                    self.full_src_tgt_dic[mr_str].append(human_reference)
                else:
                    self.full_src_tgt_dic[mr_str] = [human_reference]

            self.full_src_lst = []
            self.full_tgt_lst = []
            for src, tgt in self.full_src_tgt_dic.items():
                self.full_src_lst.append(src)
                self.full_tgt_lst.append(tgt)
            if args.dataset.use_cache:
                torch.save((self.full_src_lst, self.full_tgt_lst), cache_path)

    def __getitem__(self, index) -> T_co:
        raw_data = self.raw_datasets[index]
        raw_data.update({"struct_in": self.full_src_lst[index],
                         "text_in": "",
                         "seq_out": self.full_tgt_lst[index][0],
                         "references": self.full_tgt_lst[index]})
        return raw_data

    def __len__(self):
        return len(self.full_src_lst)
