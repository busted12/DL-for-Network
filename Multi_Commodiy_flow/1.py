import networkx as nx
fh = open('4-4')
G = nx.read_edgelist(fh)
DG = G.to_directed()
print(type(DG.edges()[0]))

