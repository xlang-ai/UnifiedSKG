import json
import argparse
import networkx as nx

from collections import defaultdict
from typing import List, Set, Dict


function_map = {'le': '<=', 'ge': '>=', 'lt': '<', 'gt': '>'}

def process_ontology(fb_roles_file, fb_types_file, reverse_properties_file):
    reverse_properties = {}
    with open(reverse_properties_file, 'r') as f:
        for line in f:
            reverse_properties[line.split('\t')[0]] = line.split('\t')[1].replace('\n', '')

    with open(fb_roles_file, 'r') as f:
        content = f.readlines()

    relation_dr = {}
    relations = set()
    for line in content:
        fields = line.split()
        relation_dr[fields[1]] = (fields[0], fields[2])
        relations.add(fields[1])

    with open(fb_types_file, 'r') as f:
        content = f.readlines()

    upper_types = defaultdict(lambda: set())

    types = set()
    for line in content:
        fields = line.split()
        upper_types[fields[0]].add(fields[2])
        types.add(fields[0])
        types.add(fields[2])

    return reverse_properties, relation_dr, relations, upper_types, types


def lisp_to_nested_expression(lisp_string: str) -> List:
    """
    Takes a logical form as a lisp string and returns a nested list representation of the lisp.
    For example, "(count (division first))" would get mapped to ['count', ['division', 'first']].
    """
    stack: List = []
    current_expression: List = []
    tokens = lisp_string.split()
    for token in tokens:
        while token[0] == '(':
            nested_expression: List = []
            current_expression.append(nested_expression)
            stack.append(current_expression)
            current_expression = nested_expression
            token = token[1:]
        current_expression.append(token.replace(')', ''))
        while token[-1] == ')':
            current_expression = stack.pop()
            token = token[:-1]
    return current_expression[0]


class SemanticMatcher:
    def __init__(self, reverse_properties, relation_dr, relations, upper_types, types):
        self.reverse_properties = reverse_properties
        self.relation_dr = relation_dr
        self.relations = relations
        self.upper_types = upper_types
        self.types = types

    def same_logical_form(self, form1, form2):
        if form1.__contains__("@@UNKNOWN@@") or form2.__contains__("@@UNKNOWN@@"):
            return False
        try:
            G1 = self.logical_form_to_graph(lisp_to_nested_expression(form1))
        except Exception:
            return False
        try:
            G2 = self.logical_form_to_graph(lisp_to_nested_expression(form2))
        except Exception:
            return False

        def node_match(n1, n2):
            if n1['id'] == n2['id'] and n1['type'] == n2['type']:
                func1 = n1.pop('function', 'none')
                func2 = n2.pop('function', 'none')
                tc1 = n1.pop('tc', 'none')
                tc2 = n2.pop('tc', 'none')

                if func1 == func2 and tc1 == tc2:
                    return True
                else:
                    return False
                # if 'function' in n1 and 'function' in n2 and n1['function'] == n2['function']:
                #     return True
                # elif 'function' not in n1 and 'function' not in n2:
                #     return True
                # else:
                #     return False
            else:
                return False

        def multi_edge_match(e1, e2):
            if len(e1) != len(e2):
                return False
            values1 = []
            values2 = []
            for v in e1.values():
                values1.append(v['relation'])
            for v in e2.values():
                values2.append(v['relation'])
            return sorted(values1) == sorted(values2)

        return nx.is_isomorphic(G1, G2, node_match=node_match, edge_match=multi_edge_match)

    def get_symbol_type(self, symbol: str) -> int:
        if symbol.__contains__('^^'):   # literals are expected to be appended with data types
            return 2
        elif symbol in self.types:
            return 3
        elif symbol in self.relations:
            return 4
        else:
            return 1

    def logical_form_to_graph(self, expression: List) -> nx.MultiGraph:
        # TODO: merge two entity node with same id. But there is no such need for
        # the second version of graphquestions
        G = self._get_graph(expression)
        G.nodes[len(G.nodes())]['question_node'] = 1
        return G

    def _get_graph(self, expression: List) -> nx.MultiGraph:  # The id of question node is always the same as the size of the graph
        if isinstance(expression, str):
            G = nx.MultiDiGraph()
            if self.get_symbol_type(expression) == 1:
                G.add_node(1, id=expression, type='entity')
            elif self.get_symbol_type(expression) == 2:
                G.add_node(1, id=expression, type='literal')
            elif self.get_symbol_type(expression) == 3:
                G.add_node(1, id=expression, type='class')
                # G.add_node(1, id="common.topic", type='class')
            elif self.get_symbol_type(expression) == 4:  # relation or attribute
                domain, rang = self.relation_dr[expression]
                G.add_node(1, id=rang, type='class')  # if it's an attribute, the type will be changed to literal in arg
                G.add_node(2, id=domain, type='class')
                G.add_edge(2, 1, relation=expression)

                if expression in self.reverse_properties:   # take care of reverse properties
                    G.add_edge(1, 2, relation=self.reverse_properties[expression])

            return G

        if expression[0] == 'R':
            G = self._get_graph(expression[1])
            size = len(G.nodes())
            mapping = {}
            for n in G.nodes():
                mapping[n] = size - n + 1
            G = nx.relabel_nodes(G, mapping)
            return G

        elif expression[0] in ['JOIN', 'le', 'ge', 'lt', 'gt']:
            G1 = self._get_graph(expression=expression[1])
            G2 = self._get_graph(expression=expression[2])
            size = len(G2.nodes())
            qn_id = size
            if G1.nodes[1]['type'] == G2.nodes[qn_id]['type'] == 'class':
                if G2.nodes[qn_id]['id'] in self.upper_types[G1.nodes[1]['id']]:
                    G2.nodes[qn_id]['id'] = G1.nodes[1]['id']
                # G2.nodes[qn_id]['id'] = G1.nodes[1]['id']
            mapping = {}
            for n in G1.nodes():
                mapping[n] = n + size - 1
            G1 = nx.relabel_nodes(G1, mapping)
            G = nx.compose(G1, G2)

            if expression[0] != 'JOIN':
                G.nodes[1]['function'] = function_map[expression[0]]

            return G

        elif expression[0] == 'AND':
            G1 = self._get_graph(expression[1])
            G2 = self._get_graph(expression[2])

            size1 = len(G1.nodes())
            size2 = len(G2.nodes())
            if G1.nodes[size1]['type'] == G2.nodes[size2]['type'] == 'class':
                G2.nodes[size2]['id'] = G1.nodes[size1]['id']
                # IIRC, in nx.compose, for the same node, its information can be overwritten by its info in the second graph
                # So here for the AND function we force it to choose the type explicitly provided in the logical form
            mapping = {}
            for n in G1.nodes():
                mapping[n] = n + size2 - 1
            G1 = nx.relabel_nodes(G1, mapping)
            G2 = nx.relabel_nodes(G2, {size2: size1 + size2 - 1})
            G = nx.compose(G1, G2)

            return G

        elif expression[0] == 'COUNT':
            G = self._get_graph(expression[1])
            size = len(G.nodes())
            G.nodes[size]['function'] = 'count'

            return G

        elif expression[0].__contains__('ARG'):
            G1 = self._get_graph(expression[1])
            size1 = len(G1.nodes())
            G2 = self._get_graph(expression[2])
            size2 = len(G2.nodes())
            # G2.nodes[1]['class'] = G2.nodes[1]['id']   # not sure whether this is needed for sparql
            G2.nodes[1]['id'] = 0
            G2.nodes[1]['type'] = 'literal'
            G2.nodes[1]['function'] = expression[0].lower()
            if G1.nodes[size1]['type'] == G2.nodes[size2]['type'] == 'class':
                G2.nodes[size2]['id'] = G1.nodes[size1]['id']

            mapping = {}
            for n in G1.nodes():
                mapping[n] = n + size2 - 1
            G1 = nx.relabel_nodes(G1, mapping)
            G2 = nx.relabel_nodes(G2, {size2: size1 + size2 - 1})
            G = nx.compose(G1, G2)

            return G

        elif expression[0] == 'TC':
            G = self._get_graph(expression[1])
            size = len(G.nodes())
            G.nodes[size]['tc'] = (expression[2], expression[3])

            return G



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='The path to dataset file for evaluation (e.g., dev.json or test.json)',
                        default=0)
    parser.add_argument('predict', type=str, default='predictions.json', help='The path to predictions')
    parser.add_argument('--fb_roles', type=str, default='fb_roles', help='The path to ontology file')
    parser.add_argument('--fb_types', type=str, default='fb_types', help='The path to ontology file')
    parser.add_argument('--reverse_properties', type=str, default='reverse_properties',
                        help='The path to ontology file')

    args = parser.parse_args()

    data_path = args.data
    with open(data_path) as f:
        data = json.load(f)
    predict_path = args.predict
    with open(predict_path) as f:
        predict = json.load(f)  # should be of format {qid: {logical_form: <str>, answer: <list>}}

    reverse_properties, relation_dr, relations, upper_types, types = process_ontology(args.fb_roles, args.fb_types,
                                                                                      args.reverse_properties)
    matcher = SemanticMatcher(reverse_properties, relation_dr, relations, upper_types, types)

    em_sum, f1_sum = 0, 0
    level_count = defaultdict(lambda : 0)
    level_em_sum = defaultdict(lambda : 0)
    level_f1_sum = defaultdict(lambda : 0)

    for item in data:
        level_count[item['level']] += 1

        answer = set()
        if item['answer'] != 'null':
            for a in item['answer']:
                answer.add(a['answer_argument'])

        if str(item['qid']) in predict:
            em = matcher.same_logical_form(predict[str(item['qid'])]['logical_form'], item['s_expression'])
            em_sum += em
            level_em_sum[item['level']] += em
            if em:
                f1_sum += 1
                level_f1_sum[item['level']] += 1
            else:
                predict_answer = set(predict[str(item['qid'])]['answer'])
                if len(predict_answer.intersection(answer)) != 0:
                    precision = len(predict_answer.intersection(answer)) / len(predict_answer)
                    recall = len(predict_answer.intersection(answer)) / len(answer)

                    f1_sum += (2 * recall * precision / (recall + precision))
                    level_f1_sum[item['level']] += (2 * recall * precision / (recall + precision))


    stats = {}
    stats['em'] = em_sum / len(data)
    stats['f1'] = f1_sum / len(data)
    stats['em_iid'] = level_em_sum['i.i.d.'] / level_count['i.i.d.']
    stats['f1_iid'] = level_f1_sum['i.i.d.'] / level_count['i.i.d.']
    stats['em_comp'] = level_em_sum['compositional'] / level_count['compositional']
    stats['f1_comp'] = level_f1_sum['compositional'] / level_count['compositional']
    stats['em_zero'] = level_em_sum['zero-shot'] / level_count['zero-shot']
    stats['f1_zero'] = level_f1_sum['zero-shot'] / level_count['zero-shot']

    print(stats)

    json.dump(predict, open("predictions.json", 'w'))
