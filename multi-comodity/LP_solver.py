from __future__ import print_function
from ortools.linear_solver import pywraplp
import numpy as np
import networkx as nx

def setVector(num_nodes, vector_3d, vector_2d):
    # convert flattened demand matrix to demand vectors
    result_vectors = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                pass
            else:
                vector_3d[i][j][i] = vector_2d[i][j]
                vector_3d[i][j][j] = -vector_2d[i][j]
                result_vectors.append(vector_3d[i][j])

    return result_vectors


def instantiate_variable(num_demand_pair,num_edge,solver,objective, cost):
    variable_matrix = np.empty(shape=(num_demand_pair, num_edge)).tolist()
    for i in range(num_demand_pair):
        for j in range(num_edge):
            variable_name = 'x' + str(i) + '_' + str(j)
            variable_matrix[i][j] = solver.IntVar(0.0, solver.infinity(), variable_name)
            objective.SetCoefficient(variable_matrix[i][j], cost)
    return variable_matrix


def set_equality_constraints(demand_vectors, nodes, solver, out_edges, in_edges, variable_matrix):
    for i, demand_vector in enumerate(demand_vectors):
        for j in nodes:

            constraintX = solver.Constraint(demand_vector[int(j)], demand_vector[int(j)])

            for out_edge_for_node_j in out_edges[int(j)]:
                constraintX.SetCoefficient(variable_matrix[i][out_edge_for_node_j], 1)

            for in_edge_for_node_j in in_edges[int(j)]:
                constraintX.SetCoefficient(variable_matrix[i][in_edge_for_node_j], -1)


def set_inequality_constraints(edges, demand_vectors, capacity, solver, variable_matrix):
    for i,edge in enumerate(edges):
        constraintX = solver.Constraint(-solver.infinity(), capacity)
        for j,demand_vector in enumerate(demand_vectors):
            constraintX.SetCoefficient(variable_matrix[j][i], 1)


def find_adjacent_edge(nodes, edges):
    '''
    you can retrieve the edge which leave node i by using out_edges[i]
    :param nodes:
    :param edges:
    :return:
    '''
    out_ind_for_all_nodes = []
    in_ind_for_all_nodes = []
    for node in nodes:
        out_ind = []
        in_ind = []
        for edge in edges:
            if edge[0] == node:
                out_ind.append(edges.index(edge))

        for edge in edges:
            if edge[1] == node:
                in_ind.append(edges.index(edge))
        out_ind_for_all_nodes.append(out_ind)
        in_ind_for_all_nodes.append(in_ind)
    return out_ind_for_all_nodes, in_ind_for_all_nodes


def load_graph(graph):
    fh = open(graph, 'rb')
    DG = nx.read_edgelist(fh,create_using=nx.DiGraph())

    nodes = sorted(DG.nodes())
    edges = sorted(DG.edges())

    return nodes, edges



def main(demand_matrix, graph='10nodes.edgelist'):
    # we are dealing here a 3 nodes graph.
    # h is a demand vector which represents the demand pair,
    # h should be 6 dimensional, where h[0] is the AB pair, h[1] is the BA pair
    # h[2] is the AC pair, h[3] is the AC pair, h[4] is the BC pair,
    # h[5] is the CB pair
    # we use edge flow representation of the solution

    # read graph

    nodes, edges = load_graph(graph)


    num_nodes = len(nodes)
    num_demand_pair = len(nodes)*(len(nodes)-1)
    num_edge = len(edges)

    # set capacity and cost
    capacity = 300
    cost = 1

    # Instantiate a mixed-integer solver, naming it SolveIntegerProblem.
    solver = pywraplp.Solver('SolveIntegerProblem',
                           pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    # Set objective
    objective = solver.Objective()

    # Instantiate variables
    variable_matrix = instantiate_variable(num_demand_pair,num_edge, solver, objective, cost)

    # Create a matrix to hold the demand matrix
    demand_matrix_flat = np.zeros((num_nodes, num_nodes, num_nodes)).tolist()

    demand_vectors = setVector(num_nodes,demand_matrix_flat,demand_matrix)

    # Find outgoing and ingoing edges for the nodes
    out_edges, in_edges = find_adjacent_edge(nodes, edges)

    # add equality constraints for each demand pair
    set_equality_constraints(demand_vectors, nodes, solver, out_edges, in_edges, variable_matrix)

    # add inequality constraints for edges
    set_inequality_constraints(edges, demand_vectors, capacity, solver, variable_matrix)

    # Solve!
    status = solver.Solve()

    if status == solver.OPTIMAL:
        edge_flow = np.zeros(num_edge)
        flag = 1
        for i in range(num_demand_pair):
            for j in range(num_edge):
                edge_flow[j] += variable_matrix[i][j].solution_value()
        print(edge_flow)

        return [flag, edge_flow]


    else:  # No optimal solution was found.
        flag = 0
        if status == solver.FEASIBLE:
            print('A potentially suboptimal solution was found.')
        else:
            print('The solver could not solve the problem.')
        return [flag]





if __name__ == '__main__':
    demand = np.random.randn((3,3,3))
    main()


    # a = 1
    # list1 = ["a","b"]
    # ArrayList list1 = new Arraylist<String>["a","b"];





