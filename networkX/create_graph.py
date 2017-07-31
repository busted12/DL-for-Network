import networkx as nx
import matplotlib.pyplot as plt


fh = open('10nodes.edgelist','rb')
G = nx.read_edgelist(fh)
DG = G.to_directed()
print(DG.edges())
