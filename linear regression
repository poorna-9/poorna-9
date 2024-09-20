import numpy as np

class LinearRegression:
    def __init__(self, X, y):
        self.x = X
        self.y = y
        self.m = 0 
        self.b = 0  

    def data_preprocessing(self):
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        minval = np.min(self.x)
        maxval = np.max(self.x)
        self.x = (self.x - minval) / (maxval - minval)

    def filling_none(self):
        mean = np.nanmean(self.x)
        self.x = np.where(np.isnan(self.x), mean, self.x)

    def fit(self, learning_rate=0.01, epochs=100):
        n = len(self.x)
        for i in range(epochs):
            y_pred = self.m * self.x + self.b
            cost = np.mean((y_pred - self.y) ** 2)

            if i % 10 == 0:
                print(f"Iteration: {i}, Cost: {cost}")

            m_gradient = (-2/n) * np.sum((self.y - y_pred) * self.x)
            b_gradient = (-2/n) * np.sum(self.y - y_pred)

            self.m -= learning_rate * m_gradient
            self.b -= learning_rate * b_gradient

    def predict(self, x):
        x = np.array(x)  
        return self.m * x + self.b
