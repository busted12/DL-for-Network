import numpy as np

learning_rate = np.linspace(0.001, 0.02, 10)
d = len(learning_rate)

for i,j in enumerate(learning_rate):
    print(i, j)