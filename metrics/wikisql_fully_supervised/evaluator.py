# encoding=utf8
import os

from . import parser4seq2seq
from .evaluation import Evaluator


class EvaluateTool(object):
    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):
        args = self.args
        summary = {}

        evaluator = Evaluator(args)

        gold_queries = [item["seq_out"] for item in golds]
        tables = [item["table"] for item in golds]

        assert len(preds) == len(gold_queries) and len(gold_queries) == len(tables)

        for table, pred, gold_query in zip(tables, preds, gold_queries):
            try:
                pred_query = parser4seq2seq.sql2query(
                    wikisql_formatted_sql_str=pred,
                    table=table,
                    lower=True,# TODO: Now it is hard code to lower letter
                )
                evaluator.add(gold_query, pred_query, table, section)
            except:
                evaluator.add_wrong()

        results = evaluator.finalize()

        summary["all_ex"] = results["total_scores"]["ex"]
        summary["all_lf"] = results["total_scores"]["lf"]
        return summary
