import os
import copy
import torch
from copy import deepcopy
from datasets import DatasetDict
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from tqdm import tqdm

def _add_adjusted_col_offsets(table):
    """Add adjusted column offsets to take into account multi-column cells."""
    adjusted_table = []
    for row in table:
        real_col_index = 0
        adjusted_row = []
        for cell in row:
            adjusted_cell = copy.deepcopy(cell)
            adjusted_cell["adjusted_col_start"] = real_col_index
            adjusted_cell["adjusted_col_end"] = (
                    adjusted_cell["adjusted_col_start"] + adjusted_cell["column_span"])
            real_col_index += adjusted_cell["column_span"]
            adjusted_row.append(adjusted_cell)
        adjusted_table.append(adjusted_row)
    return adjusted_table


def _get_heuristic_row_headers(adjusted_table, row_index, col_index):
    """Heuristic to find row headers."""
    row_headers = []
    row = adjusted_table[row_index]
    for i in range(0, col_index):
        if row[i]["is_header"]:
            row_headers.append(row[i])
    return row_headers


def _get_heuristic_col_headers(adjusted_table, row_index, col_index):
    """Heuristic to find column headers."""
    adjusted_cell = adjusted_table[row_index][col_index]
    adjusted_col_start = adjusted_cell["adjusted_col_start"]
    adjusted_col_end = adjusted_cell["adjusted_col_end"]
    col_headers = []
    for r in range(0, row_index):
        row = adjusted_table[r]
        for cell in row:
            if (cell["adjusted_col_start"] < adjusted_col_end and
                    cell["adjusted_col_end"] > adjusted_col_start):
                if cell["is_header"]:
                    col_headers.append(cell)

    return col_headers


def get_highlighted_subtable(table, cell_indices, with_heuristic_headers=False):
    """Extract out the highlighted part of a table."""
    highlighted_table = []

    adjusted_table = _add_adjusted_col_offsets(table)

    for (row_index, col_index) in cell_indices:
        cell = table[row_index][col_index]
        if with_heuristic_headers:
            row_headers = _get_heuristic_row_headers(adjusted_table, row_index,
                                                     col_index)
            col_headers = _get_heuristic_col_headers(adjusted_table, row_index,
                                                     col_index)
        else:
            row_headers = []
            col_headers = []

        highlighted_cell = {
            "cell": cell,
            "row_headers": row_headers,
            "col_headers": col_headers
        }
        highlighted_table.append(highlighted_cell)

    return highlighted_table


def linearize_subtable(subtable, table_page_title, table_section_title):
    """Linearize the highlighted subtable and return a string of its contents."""
    table_str = ""
    if table_page_title:
        table_str += "<page_title> " + table_page_title + " </page_title> "
    if table_section_title:
        table_str += "<section_title> " + table_section_title + " </section_title> "
    table_str += "<table> "

    for item in subtable:
        cell = item["cell"]
        row_headers = item["row_headers"]
        col_headers = item["col_headers"]

        # The value of the cell.
        item_str = "<cell> " + cell["value"] + " "

        # All the column headers associated with this cell.
        for col_header in col_headers:
            item_str += "<col_header> " + col_header["value"] + " </col_header> "

        # All the row headers associated with this cell.
        for row_header in row_headers:
            item_str += "<row_header> " + row_header["value"] + " </row_header> "

        item_str += "</cell> "
        table_str += item_str

    table_str += "</table>"
    return table_str


class Constructor(object):
    def __init__(self, args):
        self.args = args

    def to_seq2seq(self, raw_datasets: DatasetDict, cache_root: str):
        if not len(raw_datasets) == 2:
            raise AssertionError("Train and Dev sections of dataset expected.")
        train_dataset = TrainDataset(self.args, raw_datasets['train'], cache_root)
        dev_dataset = DevDataset(self.args, raw_datasets['validation'], cache_root)

        return train_dataset, dev_dataset


class TrainDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'totto_train.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.data = torch.load(cache_path)
        else:
            self.data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = deepcopy(raw_data)
                table_page_title = extend_data["table_page_title"]
                table_section_title = extend_data["table_section_title"]
                table = extend_data["table"]
                cell_indices = extend_data["highlighted_cells"]
                final_sentence = extend_data["final_sentences"][0]

                subtable = get_highlighted_subtable(
                        table=table,
                        cell_indices=cell_indices,
                        with_heuristic_headers=True)
                linear_table = linearize_subtable(
                        subtable=subtable,
                        table_page_title=table_page_title,
                        table_section_title=table_section_title)
                seq_out = final_sentence

                extend_data.update({"struct_in": linear_table,
                                    "text_in": "",
                                    "seq_out": seq_out})
                self.data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DevDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'totto_dev.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.data = torch.load(cache_path)
        else:
            self.data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = deepcopy(raw_data)
                table_page_title = extend_data["table_page_title"]
                table_section_title = extend_data["table_section_title"]
                table = extend_data["table"]
                cell_indices = extend_data["highlighted_cells"]
                final_sentence = extend_data["final_sentences"][0]

                subtable = get_highlighted_subtable(
                    table=table,
                    cell_indices=cell_indices,
                    with_heuristic_headers=True)
                linear_table = linearize_subtable(
                    subtable=subtable,
                    table_page_title=table_page_title,
                    table_section_title=table_section_title)
                seq_out = final_sentence

                extend_data.update({"struct_in": linear_table,
                                    "text_in": "",
                                    "seq_out": seq_out})
                self.data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def __len__(self):
        return len(self.data)
