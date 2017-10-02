import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pygraphviz as pgv
import os

dir =  u'/home/chen/MA_python/multi-comodity/Graphs/'


def draw_graph(edge_list_file, dir = u'/home/chen/MA_python/multi-comodity/Graphs/'):
    DG = nx.read_edgelist(dir + edge_list_file,create_using=nx.DiGraph())

    #DG = G.to_directed()
    #DG.remove_edge(3,2)

    A = nx.nx_agraph.to_agraph(DG)
    A.layout(prog='circo')
    A.draw(dir+ edge_list_file + '.png')

# draw_graph('4-6')
# draw_graph('4-5')
# draw_graph('4-4')
# draw_graph('4-3')
# draw_graph('3-3')
# draw_graph('3-2')
# draw_graph('4-5-2')
# draw_graph('4-6-2')
draw_graph('4-5-v3')
draw_graph('4-5-1')
draw_graph('4-5-2')
draw_graph('4-6-1')
draw_graph('4-6-2')
draw_graph('5-6')
draw_graph('6-5')
draw_graph('6-8')