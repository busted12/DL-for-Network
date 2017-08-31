from NeuralNetwork import *
from NeuralNetwork2 import *

x = u'/home/chen/MA_python/multi-comodity/Dataset/5-6-x'
y = u'/home/chen/MA_python/multi-comodity/Dataset/5-6-y'

parameter = []
n1 = NeuralNetwork(x,y)
n1.run()

n2 = NeuralNetwork2(x,y)
n2.run()

