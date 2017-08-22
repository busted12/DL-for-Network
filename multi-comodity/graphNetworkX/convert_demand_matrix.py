import numpy as np

demand_matrix = np.random.randint(1, 30,(3,3))
np.fill_diagonal(demand_matrix, 0)

print(demand_matrix)

demand_vector = np.zeros((3,3,3)).tolist()
print(demand_vector)
v=[]
for i in range(3):
    for j in range(3):
        if i==j:
            pass
        else:
            print(demand_matrix[i][j])
            demand_vector[i][j][i] = demand_matrix[i][j]
            demand_vector[i][j][j] = -demand_matrix[i][j]
            v.append(demand_vector[i][j])
print(v)

