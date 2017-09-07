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
    A.draw(edgg_list_file + '.png')

draw_graph('3-3')




