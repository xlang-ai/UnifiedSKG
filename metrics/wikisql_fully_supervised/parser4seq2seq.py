import re

from .query import Query
from .table import Table


def sql2query(wikisql_formatted_sql_str: str, table: Table, lower=True):
    """
    This function is implemented for evaluating seq2seq model's performance
    (since it predicts a "string" instead of certain format) on WikiSQL fully supervised setting.

    Assume the input string is a right formatted WikiSQL style string.
    Try to convert a WikiSQL formatted sql into Query class realized by original WikiSQL team.

    :param wikisql_formatted_sql_str:
    :param table: Table corresponding
    :param lower:
    :return:
    """
    all_regex_expression, where_suffix = r'SELECT\s+(.*)\s+FROM\s+(.*)\s*', '\s+WHERE\s+(.*)\s*'
    where_split_token = ' AND '

    agg_ops = Query.agg_ops
    cond_ops = Query.cond_ops

    header = list(table.header)  # get a header copy
    header.append('*')

    if lower:
        all_regex_expression, where_suffix = all_regex_expression.lower(), where_suffix.lower()
        where_split_token = where_split_token.lower()
        agg_ops = [agg_ops_item.lower() for agg_ops_item in agg_ops]
        cond_ops = [cond_ops_item.lower() for cond_ops_item in cond_ops]
        header = [header_item.lower() for header_item in header]

    # check whether have where clause in WikiSQL to get the right regex
    try:
        re.compile(all_regex_expression + where_suffix).match(wikisql_formatted_sql_str)
        all_regex_expression += where_suffix
    except:
        pass

    all_regex = re.compile(all_regex_expression)

    select_component, from_value, where_component = all_regex.match(wikisql_formatted_sql_str).groups()

    # handle select component, SELECT COUNT(*)... , SELECT AVG(*)...
    # get result sel: number, agg: number
    agg_op = ''
    sel_value = select_component
    for agg_op_item in agg_ops:
        agg_regex_expression = r'{}\((.*)\)'.format(agg_op_item)
        try:
            sel_value = re.compile(agg_regex_expression).match(select_component).groups()[0]
            # in case SELECT (xxx) happen
            if agg_op_item == '':
                sel_value = select_component
            agg_op = agg_op_item
        except:
            pass

    agg_index = agg_ops.index(agg_op)
    sel_index = header.index(sel_value)

    # handle where component
    conditions = []
    where_comps = where_component.split(where_split_token)
    for where_comp in where_comps:
        if not where_comp:
            break
        for i_cond_op, cond_op in enumerate(cond_ops):
            for i_header, header_item in enumerate(header):
                try:
                    prefix = '{} {} '.format(header_item, cond_op)
                    if not where_comp.startswith(prefix):
                        continue
                    where_comp_value = where_comp[len(prefix):]
                    conditions.append((i_header, i_cond_op, where_comp_value))
                except:
                    pass

    query = Query(sel_index, agg_index, conditions)

    return query