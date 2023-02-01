################################
# DO NOT EDIT THE FOLLOWING CODE
################################
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso
import numpy as np

from MyRidgeRegression import MyRidgeRegression
from my_cross_val import my_cross_val
from my_cross_val import mse

# load dataset
X, y = fetch_california_housing(return_X_y=True)

num_data, num_features = X.shape

# shuffle dataset
np.random.seed(2023)
perm = np.random.permutation(num_data)

X = X.tolist()
y = y.tolist()

X = [X[i] for i in perm]
y = [y[i] for i in perm]

X = np.array(X)
y = np.array(y)

# append column of 1s to include intercept
X = np.hstack((X, np.ones((num_data, 1))))

# Split dataset into train and test sets
NUM_TRAIN = int(np.ceil(num_data * 0.8))
NUM_TEST = num_data - NUM_TRAIN

X_train = X[:NUM_TRAIN]
X_test = X[NUM_TRAIN:]
y_train = y[:NUM_TRAIN]
y_test = y[NUM_TRAIN:]

lambda_vals = [0.01, 0.1, 1, 10, 100]

#####################
# ADD YOUR CODE BELOW
#####################

for lambda_val in lambda_vals:
    # instantiate ridge regression object
    model = MyRidgeRegression(lambda_val)

    # call to your CV function to compute mse for each fold
    errs = my_cross_val(model, 'mse', X_train, y_train)

    # print mse from CV
    print(errs)
    print("Average/SD Ridge for lambda", lambda_val, np.mean(errs), np.std(errs))

    # instantiate lasso object
    model_lasso = Lasso(alpha=lambda_val)

    # call to your CV function to compute mse for each fold
    errs_lasso = my_cross_val(model_lasso, 'mse', X_train, y_train)

    # print mse from CV
    print(errs_lasso)
    print("Average/SD Lasso for lambda", lambda_val, np.mean(errs_lasso), np.std(errs_lasso))

# instantiate ridge regression and lasso objects for best values of lambda
model = MyRidgeRegression(lambda_val=0.01)
model_lasso = Lasso(alpha=0.01)

# fit models using all training data
model.fit(X_train, y_train)
model_lasso.fit(X_train, y_train)

# predict on test data
pred = model.predict(X_test)
pred_lasso = model_lasso.predict(X_test)

# compute mse on test data
err = mse(pred, y_test)
err_lasso = mse(pred_lasso, y_test)

# print mse on test data
print("Ridge MSE: ", err)
print("Lasso MSE: ", err_lasso)
