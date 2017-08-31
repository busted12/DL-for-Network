from __future__ import print_function
from ortools.graph import pywrapgraph
import networkx as nx


def convert_list_to_edge_list(graph):
    assert type[graph] is list



def load_graph_from_edge_list(graph):
    with open(graph) as f:
        G = nx.read_edgelist(f)
        DG = G.to_directed()

    return DG


def convert_edge_list_to_list(DG):
    start_nodes = []
    end_nodes =[]
    capacities = []
    costs = []
    nodes = DG.nodes()
    edges = DG.edges(data=True)
    for i,edge in enumerate(edges):
        start_nodes.append(edges[i][0])
        end_nodes.append(edges[i][1])
        capacities.append(edges[i][2]['capacity'])
        costs.append(edges[i][2]['cost'])
    return start_nodes, end_nodes, capacities, costs

graph = 'graph.txt'
DG = load_graph_from_edge_list(graph)
start_nodes = analyze_graph(DG)
