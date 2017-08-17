from __future__ import print_function
from ortools.linear_solver import pywraplp
import numpy as np

def main(h=[20, 15, 20, 20, 25, 15]):
    # we are dealing here a 3 nodes graph.
    # h is a demand vector which represents the demand pair,
    # h should be 6 dimensional, where h[0] is the 1-2 pair, h[1] is the 2-1 pair
    # h[2] is the 1-3 pair, h[3] is the 3-1 pair, h[4] is the 2-3 pair,
    # h[5] is the 3-2 pair

    try:
        capacity = 40
        if len(h) == 6:


            # Instantiate a mixed-integer solver, naming it SolveIntegerProblem.
            solver = pywraplp.Solver('SolveIntegerProblem',
                                   pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

            # x and y are integer non-negative variables.
            x12_0 = solver.IntVar(0.0, solver.infinity(), 'x12_0')
            x12_1 = solver.IntVar(0.0, solver.infinity(), 'x12_1')
            x21_0 = solver.IntVar(0.0, solver.infinity(), 'x21_0')
            x21_1 = solver.IntVar(0.0, solver.infinity(), 'x21_1')
            x13_0 = solver.IntVar(0.0, solver.infinity(), 'x13_0')
            x13_1 = solver.IntVar(0.0, solver.infinity(), 'x13_1')
            x31_0 = solver.IntVar(0.0, solver.infinity(), 'x31_0')
            x31_1 = solver.IntVar(0.0, solver.infinity(), 'x31_1')
            x23_0 = solver.IntVar(0.0, solver.infinity(), 'x23_0')
            x23_1 = solver.IntVar(0.0, solver.infinity(), 'x23_1')
            x32_0 = solver.IntVar(0.0, solver.infinity(), 'x32_0')
            x32_1 = solver.IntVar(0.0, solver.infinity(), 'x32_1')



            # x12_0 + x12_1 = h[0]
            constraint1 = solver.Constraint(-solver.infinity(), h[0])
            constraint1.SetCoefficient(x12_0, 1)
            constraint1.SetCoefficient(x12_1, 1)

            constraint2 = solver.Constraint(-solver.infinity(), -h[0])
            constraint2.SetCoefficient(x12_0, -1)
            constraint2.SetCoefficient(x12_1, -1)

            # x21_0 + x21_1 = h[1]
            constraint3 = solver.Constraint(-solver.infinity(), h[1])
            constraint3.SetCoefficient(x21_0, 1)
            constraint3.SetCoefficient(x21_1, 1)

            constraint4 = solver.Constraint(-solver.infinity(), -h[1])
            constraint4.SetCoefficient(x21_0, -1)
            constraint4.SetCoefficient(x21_1, -1)

            # x13_0 + x13_1 = h[2]
            constraint5 = solver.Constraint(-solver.infinity(), h[2])
            constraint5.SetCoefficient(x13_0, 1)
            constraint5.SetCoefficient(x13_1, 1)

            constraint6 = solver.Constraint(-solver.infinity(), -h[2])
            constraint6.SetCoefficient(x13_0, -1)
            constraint6.SetCoefficient(x13_1, -1)

            # x31_0 + x31_1 = h[3]
            constraint7 = solver.Constraint(-solver.infinity(), h[3])
            constraint7.SetCoefficient(x31_0, 1)
            constraint7.SetCoefficient(x31_1, 1)

            constraint8 = solver.Constraint(-solver.infinity(), -h[3])
            constraint8.SetCoefficient(x31_0, -1)
            constraint8.SetCoefficient(x31_1, -1)

            # x23_0 + x23_1 = h[4]
            constraint9 = solver.Constraint(-solver.infinity(), h[4])
            constraint9.SetCoefficient(x23_0, 1)
            constraint9.SetCoefficient(x23_1, 1)

            constraint10 = solver.Constraint(-solver.infinity(), -h[4])
            constraint10.SetCoefficient(x23_0, -1)
            constraint10.SetCoefficient(x23_1, -1)

            # x32_0 + x32_1 = h[5]
            constraint11 = solver.Constraint(-solver.infinity(), h[5])
            constraint11.SetCoefficient(x32_0, 1)
            constraint11.SetCoefficient(x32_1, 1)

            constraint11 = solver.Constraint(-solver.infinity(), -h[5])
            constraint11.SetCoefficient(x32_0, -1)
            constraint11.SetCoefficient(x32_1, -1)



            # x12_0 + x21_0 + x13_1 + x31_1 + x23_1 + x32_1 <= capacity
            constraint12 = solver.Constraint(-solver.infinity(), capacity)
            constraint12.SetCoefficient(x12_0, 1)
            constraint12.SetCoefficient(x21_0, 1)
            constraint12.SetCoefficient(x13_1, 1)
            constraint12.SetCoefficient(x31_1, 1)
            constraint12.SetCoefficient(x23_1, 1)
            constraint12.SetCoefficient(x32_1, 1)

            # x13_0 + x31_0 + x23_1 + x32_1 + x12_1 + x21_1 <= capacity
            constraint13 = solver.Constraint(-solver.infinity(), capacity)
            constraint13.SetCoefficient(x13_0, 1)
            constraint13.SetCoefficient(x31_0, 1)
            constraint13.SetCoefficient(x23_1, 1)
            constraint13.SetCoefficient(x32_1, 1)
            constraint13.SetCoefficient(x12_1, 1)
            constraint13.SetCoefficient(x21_1, 1)

            # x23_0 + x32_0 + x13_1 + x31_1 + x12_1 + x21_1 <= capacity
            constraint14 = solver.Constraint(-solver.infinity(), capacity)
            constraint14.SetCoefficient(x23_0, 1)
            constraint14.SetCoefficient(x32_0, 1)
            constraint14.SetCoefficient(x12_1, 1)
            constraint14.SetCoefficient(x21_1, 1)
            constraint14.SetCoefficient(x13_1, 1)
            constraint14.SetCoefficient(x31_1, 1)

            # set link cost
            cost12 = 5
            cost13 = 5
            cost23 = 5


            # Maximize x + 10 * y.
            objective = solver.Objective()
            objective.SetCoefficient(x12_0, cost12)
            objective.SetCoefficient(x12_1, cost13 + cost23)
            objective.SetCoefficient(x21_0, cost12)
            objective.SetCoefficient(x21_1, cost13 + cost23)
            objective.SetCoefficient(x13_0, cost13)
            objective.SetCoefficient(x13_1, cost12 + cost23)
            objective.SetCoefficient(x31_0, cost13)
            objective.SetCoefficient(x31_1, cost12 + cost23)
            objective.SetCoefficient(x23_0, cost23)
            objective.SetCoefficient(x23_1, cost13 + cost12)
            objective.SetCoefficient(x32_0, cost23)
            objective.SetCoefficient(x32_1, cost13 + cost12)

            objective.SetMinimization()

            """Solve the problem and print the solution."""
            result_status = solver.Solve()
            # The problem has an optimal solution.
            assert result_status == pywraplp.Solver.OPTIMAL

            # The solution looks legit (when using solvers other than
            # GLOP_LINEAR_PROGRAMMING, verifying the solution is highly recommended!).
            assert solver.VerifySolution(1e-7, True)

            print('Number of variables =', solver.NumVariables())
            print('Number of constraints =', solver.NumConstraints())

            # The objective value of the solution.
            print('Optimal objective value = %d' % solver.Objective().Value())
            print()
            # The value of each variable in the solution.
            variable_list = [x12_0, x12_1, x13_0, x13_1, x21_0, x21_1, x23_0, x23_1,
                             x31_0, x31_1, x32_0, x32_1]

            decision_variable = np.zeros(solver.NumVariables())

            for i,variable in enumerate(variable_list):
                print('%s = %d' % (variable.name(), variable.solution_value()))
                decision_variable[i] = variable.solution_value()

            # calculate link utilization rate
            # flow_12 = x12_0 + x21_0 + x13_1 + x31_1 + x23_1 + x32_1
            flow_12 = (decision_variable[0] + decision_variable[4] + decision_variable[3] + decision_variable[9]
                      + decision_variable[7] + decision_variable[11])
            # flow 13 = x13_0 + x31_0 + x23_1 + x32_1 + x12_1 + x21_1
            flow_13 = (decision_variable[2] + decision_variable[8] + decision_variable[7] + decision_variable[1]
                      + decision_variable[11] + decision_variable[5])
            # flow 23 = x23_0 + x32_0 + x13_1 + x31_1 + x12_1 + x21_1
            flow_23 = (decision_variable[6] + decision_variable[10] + decision_variable[3] + decision_variable[1]
                      + decision_variable[9] + decision_variable[5])

            flow = [flow_12, flow_13, flow_23]

            util_12 = flow_12 / capacity
            util_13 = flow_13 / capacity
            util_23 = flow_23 / capacity

            util = [util_12, util_13, util_23]
            flag = 1
            return [flag, decision_variable, util, flow]

        else:
            raise ValueError ('h should be 6 dimensional')

    except:
        flag = 0
        print('The problem is not solvable with given constraints')
        return [flag]

if __name__ == '__main__':
   b = main()
   print('decision variabless are', b)
   #print('utilization of the edges are', util)
   #print('edge flows are', flow)