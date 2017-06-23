import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def to_nxgraph(start_nodes, end_nodes, capacity, cost):
    # start_ nodes, end_nodes, capacity, cost should be list, and have same length
    if (len(start_nodes) == len(end_nodes) == len(capacity) == len(cost)) is False:
        raise ValueError('start_ nodes, end_nodes, capacity, cost should be list, and have same length')

    else:
        num_of_edges = len(start_nodes)
        num_of_nodes = len(np.unique(start_nodes))
        # create graph
        DG = nx.Graph()

        # add nodes
        for i in range(num_of_nodes):
            DG.add_node(i)

        # add edges
        for i in range(num_of_edges):
            DG.add_edge(start_nodes[i], end_nodes[i], capacity= capacity[i], cost=cost[i])

        return DG


def to_list_form(G):
    if isinstance(G, nx.Graph) is True:
        num_of_edges = len(G.edges())
        start_nodes = np.zeros(num_of_edges)
        end_nodes = np.zeros(num_of_edges)
        capacities = np.zeros(num_of_edges)
        unit_costs = np.zeros(num_of_edges)
        for i in range(num_of_edges):
            start_nodes[i] = G.edges()[i][0]
            end_nodes[i] = G.edges()[i][1]
            capacities[i] = G.edge
        
        
    else: 
        raise TypeError('G should be a networkx graph ')


start_nodes = [0, 0, 1, 1, 1, 2, 2, 3, 4]
end_nodes = [1, 2, 2, 3, 4, 3, 4, 4, 2]
capacities = [15, 8, 20, 4, 10, 15, 5, 20, 4]
unit_costs = [4, 4, 2, 2, 6, 1, 3, 2, 3]

G = to_nxgraph(start_nodes, end_nodes, capacities, unit_costs)

print(G.edges())