import cfpq_data
import networkx

from typing import Tuple
from dataclasses import dataclass

import networkx as nx


@dataclass
class Graph:
    edges_cnt: int
    nodes_cnt: int
    labels: list


def load_graph(graph_name: str):
    path = cfpq_data.download(graph_name)
    graph = cfpq_data.graph_from_csv(path)
    edges_cnt = graph.number_of_edges()
    nodes_cnt = graph.number_of_nodes()
    labels = list(cfpq_data.get_sorted_labels(graph))
    return Graph(edges_cnt, nodes_cnt, labels)


def build_two_cycle_graph(n: int, m: int, labels: Tuple[str, str], path="./test2.dot"):
    graph = cfpq_data.labeled_two_cycles_graph(n, m, labels=labels)
    pydot_graph = networkx.drawing.nx_pydot.to_pydot(graph)
    pydot_graph.write_raw(path)


def get_graph(name: str):
    graph = cfpq_data.graph_from_csv(cfpq_data.download(name))
    return graph


def save_pydot_graph(graph, save_path: str):
    nx.drawing.nx_pydot.write_dot(graph, save_path)


def create_and_save_graph(
    cycle_1_nodes: int, cycle_2_nodes: int, labels: Tuple[str, str], save_path: str
):
    graph = cfpq_data.labeled_two_cycles_graph(
        cycle_1_nodes, cycle_2_nodes, labels=labels
    )
    save_pydot_graph(graph, save_path)
