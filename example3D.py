import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from Perceptron import Perceptron
from mpl_toolkits.mplot3d import Axes3D

# Creating a perceptron that will act in 3 dimensions
perceptron = Perceptron(3)

# Generating artificial data
data_class_1 = np.random.rand(100, 3)*1.5 + np.array([-2, 2, 2])
data_labels_1 = np.array([0]*100)

data_class_2 = np.random.rand(100, 3)*1.5 + np.array([2, -2, -2])
data_labels_2 = np.array([1]*100)

data = np.concatenate((data_class_1, data_class_2))
labels = np.concatenate((data_labels_1, data_labels_2))
data, labels = shuffle(data, labels)

# Training the perceptron
perceptron.train(data, labels)
for point in data_class_2:
    if perceptron.predict(point) == 0:
        print("Wrong")
for point in data_class_1:
    if perceptron.predict(point) == 1:
        print("Wrong")

normal = perceptron.w[1:]
b = perceptron.w[0]


x = np.arange(-3, 3, 0.1)
y = np.arange(-3, 3, 0.1)
X, Y = np.meshgrid(x, y)

Z = (b-normal[0]*X-normal[1]*Y)/normal[2]
plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(X, Y, Z, alpha=1)
plt3d.scatter(data_class_1[:, 0], data_class_1[:, 1], data_class_1[:, 2],color='orange', label='Class 1')
plt3d.scatter(data_class_2[:, 0], data_class_2[:, 1], data_class_2[:, 2],color='red', label='Class 2')
plt.title('Decision boundary separating 2 classes')
plt.legend()
plt3d.set_xlabel('Feature 1')
plt3d.set_ylabel('Feature 2')
plt3d.set_zlabel('Feature 3')
plt.show()