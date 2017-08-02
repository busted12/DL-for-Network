from __future__ import print_function
from ortools.graph import pywrapgraph
import numpy as np

def main(supply):
  """MinCostFlow simple interface example."""

  # Define four parallel arrays: start_nodes, end_nodes, capacities, and unit costs
  # between each pair. For instance, the arc from node 0 to node 1 has a
  # capacity of 15 and a unit cost of 4.

  start_nodes = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 8, 8, 8, 8, 8, 9, 9, 9]
  end_nodes = [8, 2, 4, 6, 5, 4, 7, 6, 9, 8, 9, 0, 3, 4, 2, 5, 1, 0, 2, 5, 1, 8, 3, 4, 1, 0, 8, 1, 1, 0, 9, 5, 6, 1, 8, 2]
  capacities = [23, 10, 25, 15, 17, 14, 10, 21, 17, 11, 22, 27, 14, 6, 19, 9, 11, 8, 29, 16, 22, 29, 20, 13, 18, 14, 20, 25, 13, 8, 10, 24, 5, 9, 20, 28]
  unit_costs = [6, 9, 7, 8, 8, 5, 8, 5, 6, 9, 6, 5, 6, 6, 9, 7, 8, 6, 9, 6, 5, 5, 8, 7, 5, 8, 7, 9, 7, 6, 9, 6, 5, 5, 6, 7]

  # Define an array of supplies at each node.
  supplies = supply


  # Instantiate a SimpleMinCostFlow solver.
  min_cost_flow = pywrapgraph.SimpleMinCostFlow()

  # Add each arc.
  for i in range(0, len(start_nodes)):
    min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i],
                                                capacities[i], unit_costs[i])

  # Add node supplies.

  for i in range(0, len(supplies)):
    min_cost_flow.SetNodeSupply(i, supplies[i])


  # Find the minimum cost flow between node 0 and node 4.
  if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
    print('Minimum cost:', min_cost_flow.OptimalCost())
    print('')
    print('  Arc    Flow / Capacity  Cost')
    flag = 1
    optimal_flows = np.zeros(36)
    for i in range(min_cost_flow.NumArcs()):
        cost = min_cost_flow.Flow(i) * min_cost_flow.UnitCost(i)
        print('%1s -> %1s   %3s  / %3s       %3s' % (
          min_cost_flow.Tail(i),
          min_cost_flow.Head(i),
          min_cost_flow.Flow(i),
          min_cost_flow.Capacity(i),
          cost))
        # save answer to the variable
        optimal_flows[i] = min_cost_flow.Flow(i)
    return flag, optimal_flows
  else:
    print('There was an issue with the min cost flow input.')
    flag = 0
    return flag, 0

if __name__ == '__main__':
  a, b = main([20, 0, 0, -5, -15, -30, 20, 10, 0, 0])
  print(b)