import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pygraphviz as pgv
import os

edge_list = u'/home/chen/MA_python/multi-comodity/Graphs/4-6'

G = nx.read_edgelist(edge_list)
DG = G.to_directed()

A = nx.nx_agraph.to_agraph(DG)
A.layout(prog='circo')
A.draw('file.png')




