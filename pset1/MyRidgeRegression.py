import numpy as np

class MyRidgeRegression():

    def __init__(self, lambda_val):
        self.lambda_ = lambda_val

    def fit(self, X, y):
        m, n = X.shape
        self.w_ = np.linalg.inv(X.T.dot(X) + np.identity(n) * self.lambda_).dot(X.T.dot(y))

    def predict(self, X):
        pred = X.dot(self.w_.T)
        return pred

