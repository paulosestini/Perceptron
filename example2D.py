import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from Perceptron import Perceptron

# Creating a perceptron that will act in 2 dimensions
perceptron = Perceptron(2)

# Generating artificial data
data_class_1 = np.random.rand(100, 2)*1.1 + np.array([-0.75, 0.75])
data_labels_1 = np.array([0]*100)

data_class_2 = np.random.rand(100, 2)*1.1 + np.array([0.75, -0.75])
data_labels_2 = np.array([1]*100)

data = np.concatenate((data_class_1, data_class_2))
labels = np.concatenate((data_labels_1, data_labels_2))
data, labels = shuffle(data, labels)

# Training the perceptron
perceptron.train(data, labels)

# Getting the boundary decision line
b = perceptron.w[0]
w1 = perceptron.w[1]
w2 = perceptron.w[2]

line_x0, line_x1 = -3, 3
line_y0, line_y1 = (-b-w1*line_x0)/w2, (-b-w1*line_x1)/w2

# Ploting the data and the boundary decision line
plt.scatter(data_class_1[:, 0], data_class_1[:, 1], color='orange', label='Class 1')
plt.scatter(data_class_2[:, 0], data_class_2[:, 1], color='red', label='Class 2')
plt.plot([line_x0, line_x1], [line_y0, line_y1], color='blue', label='Decision Boundary')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.title('Decision boundary separating 2 classes')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

