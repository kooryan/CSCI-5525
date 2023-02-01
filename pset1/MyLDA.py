import numpy as np

class MyLDA():

    def __init__(self, lambda_val):
        self.lambda_ = lambda_val

    def fit(self, X, y):
        c1 = X[y == 0]
        c2 = X[y == 1]

        mu_1 = np.mean(c1, axis=0)
        mu_2 = np.mean(c2, axis=0)

        s1 = np.cov(c1.T) # want it to be 2 x 2
        s2 = np.cov(c2.T)

        # calculate within-class covariance
        Sw = np.sum([s1, s2], axis=0)
        self.w = np.linalg.pinv(Sw).dot(mu_2 - mu_1).T

    def predict(self, X):
        clf = []

        x = X.dot(self.w.T)

        for i in x:
            if i >= self.lambda_:
                clf.append(1)
            else:
                clf.append(0)

        return clf

