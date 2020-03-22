import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from Perceptron import Perceptron

# Creating a 2 dimensional perceptron
perceptron = Perceptron(2)

# Creating artificial dataset
data_class_0 = np.random.rand(100, 2)*1.1 + np.array([-1, 1])
data_labels_0 = np.array([0]*100)

data_class_1 = np.random.rand(100, 2)*1.1 + np.array([1, -1])
data_labels_1 = np.array([1]*100)

data = np.concatenate((data_class_0, data_class_1))
labels = np.concatenate((data_labels_0, data_labels_1))
data, labels = shuffle(data, labels)

# Training the perceptron
perceptron.train(data, labels)

# Getting the boundary line made by the perceptron
b = perceptron.w[0]
w1 = perceptron.w[1]
w2 = perceptron.w[2]
line_x0, line_x1 = -2, 2
line_y0, line_y1 = (-b-w1*line_x0)/w2, (-b-w1*line_x1)/w2

# Plotting the data points and the boundary line
plt.scatter(data[:, 0], data[:, 1])
plt.plot([line_x0, line_x1], [line_y0, line_y1])
plt.show()

