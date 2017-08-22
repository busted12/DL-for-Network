import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pygraphviz as pgv

G = nx.Graph()

DG = G.to_directed()

DG.add_nodes_from([0,1,2,3])

DG.add_edges_from([(0,1),(0,3),(0,2),(1,3),(2,3),(2,0),(3,0),(1,0), (1,2) , (2,1), (3,2), (3,1)])



A = nx.nx_agraph.to_agraph(DG)


A.draw('file.png',prog='fdp')

