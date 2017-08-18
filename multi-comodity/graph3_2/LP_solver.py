from __future__ import print_function
from ortools.linear_solver import pywraplp
import numpy as np

def main(h=[20, 15, 20, 20, 25, 15]):
    # we are dealing here a 3 nodes graph.
    # h is a demand vector which represents the demand pair,
    # h should be 6 dimensional, where h[0] is the AB pair, h[1] is the BA pair
    # h[2] is the AC pair, h[3] is the AC pair, h[4] is the BC pair,
    # h[5] is the CB pair
    # we use edge flow representation of the solution


    capacity = 100
    cost = 1
    if len(h) == 6:


        # Instantiate a mixed-integer solver, naming it SolveIntegerProblem.
        solver = pywraplp.Solver('SolveIntegerProblem',
                               pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        num_demand_pair = 6
        num_edge = 4
        # edge0 AB, edge1 BA, edge2 BC, edge3 CB

        variable_matrix = np.empty(shape=(num_demand_pair, num_edge)).tolist()

        objective = solver.Objective()

        for i in range(num_demand_pair):
            for j in range(num_edge):
                variable_name = 'x' + str(i) + '_' + str(j)
                variable_matrix[i][j] = solver.IntVar(0.0, solver.infinity(), variable_name)
                objective.SetCoefficient(variable_matrix[i][j], cost)

        # add equality constraints

        # constraints for first demand pair AB
        # node A
        constraint1 = solver.Constraint(h[0], h[0])
        constraint1.SetCoefficient(variable_matrix[0][0], 1)
        constraint1.SetCoefficient(variable_matrix[0][1],-1)

        # node B
        constraint2 = solver.Constraint(-h[0], -h[0])
        constraint2.SetCoefficient(variable_matrix[0][0], -1)
        constraint2.SetCoefficient(variable_matrix[0][1], 1)
        constraint2.SetCoefficient(variable_matrix[0][2], 1)
        constraint2.SetCoefficient(variable_matrix[0][3], -1)

        # node C
        constraint3 = solver.Constraint(0, 0)
        constraint3.SetCoefficient(variable_matrix[0][0], 0)
        constraint3.SetCoefficient(variable_matrix[0][1], 0)
        constraint3.SetCoefficient(variable_matrix[0][2], -1)
        constraint3.SetCoefficient(variable_matrix[0][3], 1)

        # constraint for demand pair BA
        # node A
        constraint4 = solver.Constraint(-h[1], -h[1])
        constraint4.SetCoefficient(variable_matrix[1][0], 1)
        constraint4.SetCoefficient(variable_matrix[1][1], -1)
        constraint4.SetCoefficient(variable_matrix[1][2], 0)
        constraint4.SetCoefficient(variable_matrix[1][3], 0)

        # node B
        constraint5 = solver.Constraint(h[1], h[1])
        constraint5.SetCoefficient(variable_matrix[1][0], -1)
        constraint5.SetCoefficient(variable_matrix[1][1], 1)
        constraint5.SetCoefficient(variable_matrix[1][2], 1)
        constraint5.SetCoefficient(variable_matrix[1][3], -1)

        # node C
        constraint6 = solver.Constraint(0, 0)
        constraint6.SetCoefficient(variable_matrix[1][0], 0)
        constraint6.SetCoefficient(variable_matrix[1][1], 0)
        constraint6.SetCoefficient(variable_matrix[1][2], -1)
        constraint6.SetCoefficient(variable_matrix[1][3], 1)

        # constraint for demand pair AC
        # node A
        constraint7 = solver.Constraint(h[2], h[2])
        constraint7.SetCoefficient(variable_matrix[2][0], 1)
        constraint7.SetCoefficient(variable_matrix[2][1], -1)
        constraint7.SetCoefficient(variable_matrix[2][2], 0)
        constraint7.SetCoefficient(variable_matrix[2][3], 0)

        # node B
        constraint8 = solver.Constraint(0, 0)
        constraint8.SetCoefficient(variable_matrix[2][0], -1)
        constraint8.SetCoefficient(variable_matrix[2][1], 1)
        constraint8.SetCoefficient(variable_matrix[2][2], 1)
        constraint8.SetCoefficient(variable_matrix[2][3], -1)

        # node C
        constraint9 = solver.Constraint(-h[2], -h[2])
        constraint9.SetCoefficient(variable_matrix[2][0], 0)
        constraint9.SetCoefficient(variable_matrix[2][1], 0)
        constraint9.SetCoefficient(variable_matrix[2][2], -1)
        constraint9.SetCoefficient(variable_matrix[2][3], 1)

        # constraint for demand pair CA
        # node A
        constraint10 = solver.Constraint(-h[3], -h[3])
        constraint10.SetCoefficient(variable_matrix[3][0], 1)
        constraint10.SetCoefficient(variable_matrix[3][1], -1)
        constraint10.SetCoefficient(variable_matrix[3][2], 0)
        constraint10.SetCoefficient(variable_matrix[3][3], 0)

        # node B
        constraint11 = solver.Constraint(0, 0)
        constraint11.SetCoefficient(variable_matrix[3][0], -1)
        constraint11.SetCoefficient(variable_matrix[3][1], 1)
        constraint11.SetCoefficient(variable_matrix[3][2], 1)
        constraint11.SetCoefficient(variable_matrix[3][3], -1)

        # node C
        constraint12 = solver.Constraint(h[3], h[3])
        constraint12.SetCoefficient(variable_matrix[3][0], 0)
        constraint12.SetCoefficient(variable_matrix[3][1], 0)
        constraint12.SetCoefficient(variable_matrix[3][2], -1)
        constraint12.SetCoefficient(variable_matrix[3][3], 1)

        # constraint for demand pair BC
        # node A
        constraint13 = solver.Constraint(0, 0)
        constraint13.SetCoefficient(variable_matrix[4][0], 1)
        constraint13.SetCoefficient(variable_matrix[4][1], -1)
        constraint13.SetCoefficient(variable_matrix[4][2], 0)
        constraint13.SetCoefficient(variable_matrix[4][3], 0)

        # node B
        constraint14 = solver.Constraint(h[4], h[4])
        constraint14.SetCoefficient(variable_matrix[4][0], -1)
        constraint14.SetCoefficient(variable_matrix[4][1], 1)
        constraint14.SetCoefficient(variable_matrix[4][2], 1)
        constraint14.SetCoefficient(variable_matrix[4][3], -1)

        # node C
        constraint15 = solver.Constraint(-h[4], -h[4])
        constraint15.SetCoefficient(variable_matrix[4][0], 0)
        constraint15.SetCoefficient(variable_matrix[4][1], 0)
        constraint15.SetCoefficient(variable_matrix[4][2], -1)
        constraint15.SetCoefficient(variable_matrix[4][3], 1)


        # constraint for demand pair CB
        # node A
        constraint16 = solver.Constraint(0, 0)
        constraint16.SetCoefficient(variable_matrix[5][0], 1)
        constraint16.SetCoefficient(variable_matrix[5][1], -1)
        constraint16.SetCoefficient(variable_matrix[5][2], 0)
        constraint16.SetCoefficient(variable_matrix[5][3], 0)

        # node B
        constraint17 = solver.Constraint(-h[5], -h[5])
        constraint17.SetCoefficient(variable_matrix[5][0], -1)
        constraint17.SetCoefficient(variable_matrix[5][1], 1)
        constraint17.SetCoefficient(variable_matrix[5][2], 1)
        constraint17.SetCoefficient(variable_matrix[5][3], -1)

        # node C
        constraint18 = solver.Constraint(h[5], h[5])
        constraint18.SetCoefficient(variable_matrix[5][0], 0)
        constraint18.SetCoefficient(variable_matrix[5][1], 0)
        constraint18.SetCoefficient(variable_matrix[5][2], -1)
        constraint18.SetCoefficient(variable_matrix[5][3], 1)

        # capacity constraint for edge 0
        constraint19 = solver.Constraint(-solver.infinity(), capacity)
        constraint19.SetCoefficient(variable_matrix[0][0], 1)
        constraint19.SetCoefficient(variable_matrix[1][0], 1)
        constraint19.SetCoefficient(variable_matrix[2][0], 1)
        constraint19.SetCoefficient(variable_matrix[3][0], 1)
        constraint19.SetCoefficient(variable_matrix[4][0], 1)
        constraint19.SetCoefficient(variable_matrix[5][0], 1)

        # capacity constraint for edge 1
        constraint20 = solver.Constraint(-solver.infinity(), capacity)
        constraint20.SetCoefficient(variable_matrix[0][1], 1)
        constraint20.SetCoefficient(variable_matrix[1][1], 1)
        constraint20.SetCoefficient(variable_matrix[2][1], 1)
        constraint20.SetCoefficient(variable_matrix[3][1], 1)
        constraint20.SetCoefficient(variable_matrix[4][1], 1)
        constraint20.SetCoefficient(variable_matrix[5][1], 1)

        # capacity constraint for edge 2
        constraint21 = solver.Constraint(-solver.infinity(), capacity)
        constraint21.SetCoefficient(variable_matrix[0][2], 1)
        constraint21.SetCoefficient(variable_matrix[1][2], 1)
        constraint21.SetCoefficient(variable_matrix[2][2], 1)
        constraint21.SetCoefficient(variable_matrix[3][2], 1)
        constraint21.SetCoefficient(variable_matrix[4][2], 1)
        constraint21.SetCoefficient(variable_matrix[5][2], 1)

        # capacity constraint for edge 3
        constraint22 = solver.Constraint(-solver.infinity(), capacity)
        constraint22.SetCoefficient(variable_matrix[0][3], 1)
        constraint22.SetCoefficient(variable_matrix[1][3], 1)
        constraint22.SetCoefficient(variable_matrix[2][3], 1)
        constraint22.SetCoefficient(variable_matrix[3][3], 1)
        constraint22.SetCoefficient(variable_matrix[4][3], 1)
        constraint22.SetCoefficient(variable_matrix[5][3], 1)


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
    main()






