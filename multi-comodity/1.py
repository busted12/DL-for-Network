import networkx as nx
import pygraphviz as pgv # need pygraphviz or pydot for nx.to_agraph()

G = nx.DiGraph()
G.add_edge(1,2,weight=7)
G.add_edge(2,3,weight=8)
G.add_edge(3,4,weight=1)
G.add_edge(4,1,weight=11)
G.add_edge(1,3)
G.add_edge(2,4)

for u,v,d in G.edges(data=True):
    d['label'] = d.get('weight','')

A = nx.nx_agraph.to_agraph(G)
A.layout(prog='dot')
A.draw('test.png')