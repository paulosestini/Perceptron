import numpy as np

class Perceptron:
    def __init__(self, dim):
        # The dimension of the feature space
        self.dim = dim
        
        # Weights (the first weight is the bias)
        # W = [w0, w1, ..., wn] = [b, w1, ..., wn]
        self.w = np.random.rand(dim+1)
        
        self.lrate = 0.1 # Learning rate

    # Step in the training process
    def __step(self, x, y):
        # Appending 1 to the input vector
        # [x1, x2, ..., xn] -> [1, x1, ..., xn]
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
        # Appending 1 to the input vector
        # [x1, x2, ..., xn] -> [1, x1, ..., xn]
        x = np.array((1, *x))
        
        out = self.w.dot(x)
        
        if out > 0:
            return 0
        else:
            return 1
        


