#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adjust from the RAT_SQL, the same with original WikiSQL

import numpy as np

from . import dbengine, query, table


class Evaluator:
    """
    An evaluator similar with Spider one.
    """
    def __init__(self, args):
        self.args = args
        db_train_path, db_dev_path, db_test_path = args.wikisql.db_path.split(' ')
        self.db_engines = {"train": dbengine.DBEngine(db_train_path), "dev": dbengine.DBEngine(db_dev_path), "test": dbengine.DBEngine(db_test_path) }
        self.lf_match = []
        self.exec_match = []

    def __evaluate_one(self, gold_query: query.Query, inferred_code: query.Query, table: table.Table, section):
        db_engine = self.db_engines[section]
        gold_result = db_engine.execute_query(table.table_id, gold_query, lower=True)
        # result is a list contain values. e.g. ['united states']
        pred_query = None
        pred_result = None
        try:
            pred_query = inferred_code
            pred_result = db_engine.execute_query(table.table_id, pred_query, lower=True)
        except:
            pass

        lf_match = gold_query == pred_query
        exec_match = gold_result == pred_result

        return lf_match, exec_match

    def add(self, gold_query: query.Query, inferred_code: query.Query, table: table.Table, section, orig_question=None):
        lf_match, exec_match = self.__evaluate_one(gold_query, inferred_code, table, section)
        self.lf_match.append(lf_match)
        self.exec_match.append(exec_match)

    def add_wrong(self):
        self.lf_match.append(False)
        self.exec_match.append(False)

    def finalize(self):
        mean_exec_match = float(np.mean(self.exec_match))
        mean_lf_match = float(np.mean(self.lf_match))

        return {
            'per_item': [{'ex': ex, 'lf': lf} for ex, lf in zip(self.exec_match, self.lf_match)],
            'total_scores': {'ex': mean_exec_match, 'lf': mean_lf_match},
        }
