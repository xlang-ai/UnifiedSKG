import csv
from collections import defaultdict
from .APIs import *
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
sw = stopwords.words('english')
order_dict = [r'zero', r'first', r'second',
              r'third', r'fourth', r'fifth',
              r'sixth', r'seventh', r'eighth',
              r'ninth', r'tenth']
number_dict = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'nine',
               '9': 'nine', '10': 'ten'}

def load_data():
    reader = csv.reader(open("logic2text_labeled.csv", encoding='utf-8'))
    data = []
    for i, row in enumerate(reader):
        if i == 0: continue
        data.append(row)
    return data


def digit_match(x, nl):
    found = 0
    if x in nl:
        found = 1
    elif x == '1':
        found = 1
    elif int(x) <= 10:
        if re.search(order_dict[int(x)], nl) or re.search(number_dict[str(int(x))], nl):
            found = 1
    elif format(int(x), ',') in nl or x in nl:
        found = 1
    return found


def logic_matching(logic, nl, truth=None):

    logic = logic.replace('{', '|')
    logic = logic.replace('}', '|').replace(';', '|')
    logic = [x.strip() for x in logic.split('|') if len(x.strip())]
    processed_logic = []
    if 'not' in nl and 'not' not in logic: processed_logic.append('not')
    # In this version, we only add "not" as the reversed judgement
    for x in logic:
        found = 0
        if x in APIs.keys():
            if 'alias' in APIs[x].keys():
                for regex in APIs[x]['alias']:
                    if re.search(regex, nl) is not None:
                        found = 1
                        break
            else:
                found = 1
        elif x.isdigit():
            if digit_match(x, nl):
                found = 1
            elif truth is not None and not digit_match(x, truth):
                found = 1
        else:
            found = 1
        if found == 0:
            processed_logic.append(x)
    return processed_logic


def count_label(data):
    count = defaultdict(int)
    labels = [x[-1] for x in data]
    for label in labels:
        count[label] += 1
    print(count)
