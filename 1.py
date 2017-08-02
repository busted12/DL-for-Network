import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

d2 = json.load(open("result_M3.txt"))

accuracy = d2['accuracy']
metric2 = d2['metric2']
metric3 = d2['metric3']

plt.scatter(metric2,metric3)
plt.xlabel('M2')
plt.ylabel('M3')
plt.show()