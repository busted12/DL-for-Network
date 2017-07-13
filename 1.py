import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


a = [1, 2, 3, 4]
b = [4,3,3,1]

counter= 0
for i,j in zip(a, b):
    if i == j:
        counter +=1
print(counter)
