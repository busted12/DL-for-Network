from __future__ import print_function
from ortools.linear_solver import pywraplp
import networkx as nx
import numpy as np



def main(graph, flow, ca):

    nodes, edges = read_graph(graph)

    out_edges, in_edges = find_adjacent_edges(nodes, edges)

    add_constraints()

    solver = pywraplp.Solver('SolveIntegerProblem',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    objective = solver.Objective()

    pass


def read_graph(graph):
    fh = open(graph)
    G = nx.read_edgelist(fh)
    DG = G.to_directed()
    nodes = sorted(DG.nodes())
    edges = sorted(DG.edges())

    return nodes, edges


def instantiate_variable(edges):
    variable_vector = np.empty(shape=(len(edges), 1)).tolist()
    for i, variable in enumerate(variable_vector):
        variable_name = 'x' + str(i)



def add_constraints():
    add_equality_constraints()
    add_inequality_constraints()
    pass


def find_adjacent_edges(nodes, edges):
    out_going_edges = []
    in_going_edges = []
    for node in nodes:
        out_edge = []
        in_edge = []
        for edge in edges():
            if edge[0] == node:
                out_edge.append(edges.index(edge))
            else:
                pass
            if edge[1] == node:
                in_edge.append(edges.index(edge))
            else:
                pass
        out_going_edges.append(out_edge)
        in_going_edges.append(in_edge)
    return out_going_edges, in_going_edges


def add_equality_constraints(nodes, demand_vector, solver, out_edges, in_edges):
    for node in nodes:
        constraintX = solver.Constraint(demand_vector[node],demand_vector[node])
        for out_edge in out_edges[nodes]:
            constraintX.SetCoefficient(out_edge, 1)
        for in_edge in in_edges[nodes]:
            constraintX.SetCoefficient(in_edge, -1)

def add_inequality_constraints(edges,solver, capacity):
    for edge in edges:
        constraintX = solver.Constraint(-solver.infinity(), capacity)
        constraintX.SetCoefficient()
