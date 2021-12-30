import os
import torch
from datasets import DatasetDict
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import numpy as np
import copy
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
    "qid": datasets.Value("string"),
    "question": datasets.Value("string"),
    "answer": datasets.features.Sequence(
        {
            "answer_type": datasets.Value("string"),
            "answer_argument": datasets.Value("string"),
            "entity_name": datasets.Value("string"),
        }
    ),
    "function": datasets.Value("string"),
    "num_node": datasets.Value("int32"),
    "num_edge": datasets.Value("int32"),
    "graph_query": {
        "nodes": datasets.features.Sequence(
            {
                "nid": datasets.Value("int32"),
                "node_type": datasets.Value("string"),
                "id": datasets.Value("string"),
                "class": datasets.Value("string"),
                "friendly_name": datasets.Value("string"),
                "question_node": datasets.Value("int32"),
                "function": datasets.Value("string"),
            }
        ),
        "edges": datasets.features.Sequence(
            {
                "start": datasets.Value("int32"),
                "end": datasets.Value("int32"),
                "relation": datasets.Value("string"),
                "friendly_name": datasets.Value("string"),
            }
        ),
    },
    "sparql_query": datasets.Value("string"),
    "domains": datasets.features.Sequence(datasets.Value("string")),
    "level": datasets.Value("string"),
    "s_expression": datasets.Value("string"),
    "retrieved_schema": datasets.features.Sequence(datasets.Value("string"))
}
"""


def grailqa_get_entity(raw_data: dict) -> list:
    nodes = raw_data["graph_query"]["nodes"]

    entities = []
    for type, id, name in zip(nodes["node_type"], nodes["id"], nodes["friendly_name"]):
        if type == "entity":
            entities.append([name.lower(), id])

    return entities


def grailqa_get_schema(raw_data: dict, max_count: int = 20) -> list:
    nodes = raw_data["graph_query"]["nodes"]
    edges = raw_data["graph_query"]["edges"]

    identified_schema = set()
    for type, id, is_question_node in zip(nodes["node_type"], nodes["id"], nodes["question_node"]):
        if type == "class" and is_question_node:
            identified_schema.add(id)

    for rel in edges["relation"]:
        identified_schema.add(rel)

    other_retrieved_schema = list(set(raw_data["retrieved_schema"]) - identified_schema)
    np.random.shuffle(other_retrieved_schema)

    schema = list(identified_schema) + other_retrieved_schema[: max(0, max_count - len(identified_schema))]
    np.random.shuffle(schema)

    return schema


def grailqa_get_input(question: str, entity: list, schema: list):
    serialized_entity = " ".join([": ".join(elm) for elm in entity])
    serialized_schema = " ".join(schema)
    return question.strip(), serialized_entity.strip() + " | " + serialized_schema.strip()


class TrainDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'grailqa_train.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.extended_data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = copy.deepcopy(raw_data)
                question = raw_data["question"]
                entity = grailqa_get_entity(raw_data)
                schema = grailqa_get_schema(raw_data)
                text_in, struct_in = grailqa_get_input(question, entity, schema)
                seq_out = raw_data["s_expression"]

                extend_data.update({"text_in": text_in, "struct_in": struct_in, "seq_out": seq_out})
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

        cache_path = os.path.join(cache_root, 'grailqa_dev.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.extended_data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = copy.deepcopy(raw_data)
                question = raw_data["question"]
                entity = grailqa_get_entity(raw_data)
                schema = grailqa_get_schema(raw_data)
                text_in, struct_in = grailqa_get_input(question, entity, schema)
                seq_out = raw_data["s_expression"]

                extend_data.update({"text_in": text_in, "struct_in": struct_in, "seq_out": seq_out})
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

        cache_path = os.path.join(cache_root, 'grailqa_test.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.extended_data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = copy.deepcopy(raw_data)
                question = raw_data["question"]
                entity = grailqa_get_entity(raw_data)
                schema = grailqa_get_schema(raw_data)
                text_in, struct_in = grailqa_get_input(question, entity, schema)
                seq_out = raw_data["s_expression"]

                extend_data.update({"text_in": text_in, "struct_in": struct_in, "seq_out": seq_out})
                self.extended_data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)
