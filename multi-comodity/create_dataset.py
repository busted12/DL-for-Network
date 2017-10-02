from LP_solver import *
from Vlink_Reconf.Vlink.Toolset import *


def create_datset(number_of_trials, graph):
    '''
    feed the solver with number_of_trials randomly generated arrays
    :param number_of_trials:
    :return:
    '''
    fh = open(graph, 'rb')
    DG = nx.read_edgelist(fh, create_using=nx.DiGraph())

    nodes = sorted(DG.nodes())
    edges = sorted(DG.edges())

    num_nodes = len(nodes)
    num_demand_pair = len(nodes)*(len(nodes)-1)
    num_edges = len(edges)

    counter = 0
    x_data = np.zeros(num_demand_pair)
    y_data = np.zeros(num_edges)
    for i in range(number_of_trials):
        demand_matrix = np.random.randint(10, 50, (num_nodes, num_nodes))
        np.fill_diagonal(demand_matrix, 0)

        flat_demand_matrix = demand_matrix.ravel()
        demand_vector = flat_demand_matrix[np.nonzero(flat_demand_matrix)]

        d = main(demand_matrix, graph)
        if d[0] == 1:
            counter = counter + 1
            x_data = np.vstack((x_data, demand_vector))
            y_data = np.vstack((y_data, d[1]))
            print counter

    return x_data[1:counter+1, ], y_data[1:counter+1, ]

graph = '4-5-v3'

graph_root = u'/home/chen/MA_python/multi-comodity/Graphs/'
graph_dir = graph_root + graph
x, y = create_datset(40000, graph_dir)
x_new, y_new = remove_dup_dataset(x, y)

file_root = u'/home/chen/MA_python/multi-comodity/Dataset/'
x_data_dir = file_root + graph + '-x'
y_data_dir = file_root + graph + '-y'

np.savetxt(x_data_dir, x_new)
np.savetxt(y_data_dir, y_new)
