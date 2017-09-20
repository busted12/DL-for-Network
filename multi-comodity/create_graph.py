import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pygraphviz as pgv
import os

dir =  u'/home/chen/MA_python/multi-comodity/Graphs/'
graph = u'4-6'


def draw_graph(edgg_list_file, dir = u'/home/chen/MA_python/multi-comodity/Graphs/'):
    G = nx.read_edgelist(dir + edgg_list_file)
    DG = G.to_directed()

    A = nx.nx_agraph.to_agraph(DG)
    A.layout(prog='circo')
    A.draw(dir+ edgg_list_file + '.pdf')

draw_graph('4-6')
draw_graph('4-5')
draw_graph('4-4')
draw_graph('4-3')
draw_graph('3-3')
draw_graph('3-2')



