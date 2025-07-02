import random
import pandas as pd

class LinearRegression:
    def __init__(self, X_train, y_train, learning_rate=0.01):
        self.X_train, self.y_train = X_train, y_train
        self.weight, self.bias = 0, 0
        self.lr = learning_rate
    
    def train(self, epochs):
        n = len(self.X_train)
        for _ in range(epochs):
            y_pred = self.weight * self.X_train + self.bias
            error = y_pred - self.y_train

            dw = (2 / n) * (error * self.X_train).sum()
            db = (2 / n) * error.sum()

            self.weight -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X_test):
        return self.weight * X_test + self.bias

