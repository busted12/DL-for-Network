import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

d2 = json.load(open("result_net_structure.txt"))

accuracy = d2['accuracy']
metric2 = d2['metric2']

print(len(accuracy[0:500]))

print(len(metric2[0:500]))

plt.scatter(metric2,accuracy)
plt.xlabel('M2')
plt.ylabel('accuracy')
plt.show()