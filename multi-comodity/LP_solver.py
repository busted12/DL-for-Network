from __future__ import print_function
from ortools.linear_solver import pywraplp
import numpy as np
import networkx as nx




def main(demand_matrix=[[0 ,17 ,26],[21,0 ,11],[14, 11, 0]], graph='10nodes.edgelist'):
    # we are dealing here a 3 nodes graph.
    # h is a demand vector which represents the demand pair,
    # h should be 6 dimensional, where h[0] is the AB pair, h[1] is the BA pair
    # h[2] is the AC pair, h[3] is the AC pair, h[4] is the BC pair,
    # h[5] is the CB pair
    # we use edge flow representation of the solution


    fh = open(graph, 'rb')
    G = nx.read_edgelist(fh)
    DG = G.to_directed()

    nodes = sorted(DG.nodes())
    edges = sorted(DG.edges())

    num_nodes = len(nodes)
    num_demand_pair = len(nodes)*(len(nodes)-1)
    num_edge = len(edges)

    capacity = 300
    cost = 1



    # Instantiate a mixed-integer solver, naming it SolveIntegerProblem.
    solver = pywraplp.Solver('SolveIntegerProblem',
                           pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)


    # edge0 AB, edge1 BA, edge2 BC, edge3 CB

    demand_matrix_flat = np.zeros((num_nodes, num_nodes, num_nodes)).tolist()

    # convert flattened demand matrix to demand vectors
    demand_vectors = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                pass
            else:
                demand_matrix_flat[i][j][i] = demand_matrix[i][j]
                demand_matrix_flat[i][j][j] = -demand_matrix[i][j]
                demand_vectors.append(demand_matrix_flat[i][j])

    variable_matrix = np.empty(shape=(num_demand_pair, num_edge)).tolist()

    objective = solver.Objective()

    out_edges, in_edges = find_adjacent_edge(nodes, edges)
    #print(out_edges)


    # instantiate variables
    for i in range(num_demand_pair):
        for j in range(num_edge):
            variable_name = 'x' + str(i) + '_' + str(j)
            variable_matrix[i][j] = solver.IntVar(0.0, solver.infinity(), variable_name)
            objective.SetCoefficient(variable_matrix[i][j], cost)

        '''
                for i in range(num_demand_pair):
                  for j in range(graph.vertices):
                      constraintX = solver.Constraint(dict())
                      for k in range(i.incidentEdges):
                          constraintX.SetCoefficient(variable_matrix[j][k.target], 1)
                      for k in range(i.outgoingEdges):
                          constraintX.SetCoefficient(variable_matrix[j][k.source], -1)
        '''

    # add equality constraints for each demand pair
    for i, demand_vector in enumerate(demand_vectors):
        #print('the demand vector for %s th demand pair is %s' % (i, demand_vector))
        for j in nodes:
            constraintX = solver.Constraint(demand_vector[int(j)], demand_vector[int(j)])
            #print('this is constraint for %s th node' % j)
            #print(out_edges[int(j)])

            for out_edge_for_node_j in out_edges[int(j)]:
                #print('outgoing edges for node %s are %s' % (j, out_edge_for_node_j))
                constraintX.SetCoefficient(variable_matrix[i][out_edge_for_node_j], 1)

            for in_edge_for_node_j in in_edges[int(j)]:
                constraintX.SetCoefficient(variable_matrix[i][in_edge_for_node_j], -1)


    # add inequality constraints for edges
    for i,edge in enumerate(edges):
        constraintX = solver.Constraint(-solver.infinity(), capacity)
        for j,demand_vector in enumerate(demand_vectors):
            constraintX.SetCoefficient(variable_matrix[j][i], 1)



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


if __name__ == '__main__':
    main()






