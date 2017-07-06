import numpy as np
from min_cost_flow import main

'''
a = [1, 2, 3]
b = np.array([1, 2, 3])

print(a)
print(b)
c = b.tolist()
print(c)
'''


def generate_data(number_of_inputs):
    x_data = np.zeros(5)
    y_data = np.zeros(9)
    counter = 0
    for i in range(number_of_inputs):
        # a is a random vector with integers ranging from -20 to 20
        initial_array = np.random.randint(low=-30, high=30, size=4)

        # missing_element is the minus of sum of all element in initial_array.
        missing_element = np.sum(initial_array)

        # balanced_array is the vector of 5 dimensions, the sum of which is zero
        balanced_array = (np.append(initial_array, -missing_element))

        # shuffle the array to make it random
        np.random.shuffle(balanced_array)

        # convert the array to list to serve as input for MIN_Flow_Solver
        shuffled_list = balanced_array.tolist()

        (flag, output) = main(shuffled_list)
        if flag == 1:
            counter = counter + 1
            x_data = np.vstack((x_data, balanced_array))
            #print(flag)
            #print(output)
            y_data = np.vstack((y_data, output))

    print(counter)
    #print(d)

    #print(k[1:counter+1, ])
    #print(d[1:counter+1, ])
    return x_data[1:counter+1, ], y_data[1:counter+1], counter

x = generate_data(50000000)
np.savetxt('x_data_set2', x[0])
np.savetxt('y_data_set2', x[1])
print(x[2])
