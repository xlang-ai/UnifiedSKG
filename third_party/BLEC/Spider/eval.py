import re


number_dict = {'2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six'}

agg_dict = {'count': ['more', 'number', 'how many', 'most', 'one', 'at least', 'only', 'more than', 'fewer than'],
           'avg': ['average', 'mean'],
           'max': ['maximum', 'largest', 'longest', 'oldest', 'top', 'best', 'most', 'highest', 'latest', 'larger than any', 'lowest rank','predominantly'],
           'min':['minimum', 'smallest', 'shortest', 'worst', 'youngest', 'least', 'lowest', 'earliest', 'any', 'highest rank']}
sc_dict = {'asc': ['least', 'smallest', 'lowest', 'ascending', 'fewest', 'alphabetical order','lexicographical order'],
           'desc': ['most', 'largest', 'highest', 'descending', 'greatest','reverse alphabetical order','reversed lexicographical order']}

op_dict = {'>': ['more than', 'older than', 'bigger than', 'larger than', 'higher than', 'more', 'after', 'greater', 'above', 'over', 'at least'],
           '<': ['less than', 'fewer than', 'younger than', 'smaller than', 'lower than', 'less', 'before', 'shorter', 'below', 'under', 'lighter'],}


def question_test(sql, name_dict, question):
    label = 1
    error = []
    # not issues
    if ('not' in question or 'n\'t' in question or 'without' in question) and \
            '!=' not in name_dict.values() and \
            'except' not in sql.lower() and 'not' not in sql.lower():
        label = 0
        error.append('not')

    # reverse number
    if any(y in question and x not in sql for x, y in number_dict.items()):
        label -= 1
        error.append('number')

    # DISTINCT
    if 'distinct' in sql.lower() and ('different' not in question or 'distinct' not in question):
        # doesn't matters in spider dataset
        ...

    # components issues
    for key, val in name_dict.items():

        # agg issues
        if 'AGG' in key:
            for agg in agg_dict:
                if agg in val and not any(x in question for x in agg_dict[agg]):
                    label -= 1
                    error.append(agg)

        # sc issues
        if 'SC' in key:
            val = val.lower()
            for sc in sc_dict:
                if sc in val and not any(x in question for x in sc_dict[sc]):
                    label -= 1
                    error.append(sc)

        # op issues
        if 'OP' in key:
            for op in op_dict:
                if op == val and not any(x in question for x in op_dict[op]):
                    label -= 1
                    error.append(op)
        # deal with value
        if 'VALUE' in key:
            if val == '1' or val == 't':
                continue
            if val in number_dict:
                tem = number_dict[val]
                if re.search(r"{}".format(tem), question):
                    question = re.sub(r"{}".format(tem), '__FOUND__', question)
                    continue
            val = val.strip('\'').strip('\"')
            if re.search(r"{}".format(val), question):
                question = re.sub(r"{}".format(val), '__FOUND__', question)
            else:
                label -= 1
                error.append(val)

        # deal with columns before FROM
        if 'COLUMN' in key:
            # not useful
            # pos = sql.lower().find(val)
            # pos_from = sql.lower().find('from')
            # if pos<pos_from:
            #     val = val.split('_')
            #     if any(x not in question for x in val):
            #         label -= 1
            #         error.append('_'.join(val))
            ...

    return label, error
