import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

fh = open('10nodes.edgelist','rb')
G = nx.read_edgelist(fh)
DG = G.to_directed()

#nx.set_edge_attributes(DG, 'capacity', 10)
#nx.set_edge_attributes(DG, 'unit cost', 10)

np.random.seed(1)
# iterate edges and assign attributes
for n in DG.edges_iter():
    DG.edge[n[0]][n[1]]['unit cost'] = np.random.randint(low=5, high=10)
    DG.edge[n[0]][n[1]]['capacity'] = np.random.randint(low=5, high=30)

# create list to represent the graph#
'''  example
  start_nodes = [0, 0, 1, 1, 1, 2, 2, 3, 4]
  end_nodes = [1, 2, 2, 3, 4, 3, 4, 4, 2]
  capacities = [15, 8, 20, 4, 10, 15, 5, 20, 4]
  unit_costs = [4, 4, 2, 2, 6, 1, 3, 2, 3]
'''

start_nodes = []
end_nodes = []
capacities = []
unit_costs = []

for i in range(len(DG.edge)):
    for key in DG.edge[str(i)]:
        start_nodes.append(i)
        end_nodes.append(int(key))
        capacities.append(DG.edge[str(i)][key]['capacity'])
        unit_costs.append(DG.edge[str(i)][key]['unit cost'])



nodes = sorted(DG.nodes())
edges = sorted(DG.edges())
print(nodes)
print(edges)


def find_adjacent_edge(nodes, edges):
    out_ind_for_all_nodes = []
    in_ind_for_all_nodes = []
    for node in nodes:
        out_ind = []
        in_ind = []
        for edge in edges:
            if edge[0] == node:
                out_ind.append(edges.index(edge))

        for edge in edges:
            if edge[1] == node:
                in_ind.append(edges.index(edge))
        out_ind_for_all_nodes.append(out_ind)
        in_ind_for_all_nodes.append(in_ind)
    return out_ind_for_all_nodes, in_ind_for_all_nodes

out_edges, in_edges = find_adjacent_edge(nodes,edges)

print(out_edges)
for i in nodes:
    print(i)
    print(out_edges[int(i)])

for i in nodes:
    print(i)
    print(in_edges[int(i)])
