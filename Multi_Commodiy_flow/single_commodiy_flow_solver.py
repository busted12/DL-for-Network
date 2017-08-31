from __future__ import print_function
from ortools.linear_solver import pywraplp
import networkx as nx


class SingleCommoditySolver(object):
    def __init__(self, graph, flow):
        self.graph = graph
        self.flow = flow

    def solve(self):
        SingleCommoditySolver.read_graph()
        SingleCommoditySolver.add_constraints()
        solver = pywraplp.Solver('SolveIntegerProblem',
                                 pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        objective = solver.Objective()

    def instantiate_variable(self):




        pass

    def read_graph(self):
        fh = open(self.graph)
        G = nx.read_edgelist(fh)
        DG = G.to_directed()
        nodes = sorted(DG.nodes())
        edges = sorted(DG.edges())

        return nodes, edges

    def find_adjacent_edges(self):
        for edge in edges:
            if





    def add_constraints(self):


        pass

    def add_equality_constraints(self):

        pass

    def add_inequality_constraints(self):
        pass
