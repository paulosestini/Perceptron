import numpy as np

class Perceptron:
    def __init__(self, dim):
        self.dim = dim
        self.w = np.random.rand(dim+1)
        self.lrate = 0.1

    def __step(self, x, y):
        x = np.array((1, *x))
        out = self.w.dot(x)

        if out > 0 and y == 1:
            self.w = self.w - self.lrate*x
        elif out <= 0 and y == 0:
            self.w = self.w + self.lrate*x
    
    def train(self, data, labels, iters = 100):
        for _ in range(iters):
            for (x, y) in zip(data, labels):
                self.__step(x, y)
    
    def predict(self, x):
        x = np.array((1, *x))
        out = self.w.dot(x)
        if out > 0:
            return 0
        else:
            return 1
        


