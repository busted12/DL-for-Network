import numpy as np
from random import randint
from min_cost_flow import main1

'''
d = np.array([1,2, 3])
for i in range(5):
    a1 = np.random.rand(1, 3)
    d = np.vstack((d,a1))

print(d)
'''

d = np.array([[1,2],[2,3]])
print(d)

c = d[0, :]
print(c)
