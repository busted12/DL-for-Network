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


print(start_nodes)
print(end_nodes)
print(capacities)
print(unit_costs)

