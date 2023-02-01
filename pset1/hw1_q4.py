################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from my_cross_val import my_cross_val
from MyLDA import MyLDA
from my_cross_val import err_rate

# load dataset
data = pd.read_csv('hw1_q4_dataset.csv', header=None).to_numpy()
X = data[:,:-1]
y = data[:,-1]

num_data, num_features = X.shape

plt.scatter(X[:1000, 0], X[:1000, 1])
plt.scatter(X[1000:, 0], X[1000:, 1])
plt.show()

# shuffle dataset
np.random.seed(2023)
perm = np.random.permutation(num_data)

X = X.tolist()
y = y.tolist()

X = [X[i] for i in perm]
y = [y[i] for i in perm]

X = np.array(X)
y = np.array(y)

# Split dataset into train and test sets
NUM_TRAIN = int(np.ceil(num_data*0.8))
NUM_TEST = num_data - NUM_TRAIN

X_train = X[:NUM_TRAIN]
X_test = X[NUM_TRAIN:]
y_train = y[:NUM_TRAIN]
y_test = y[NUM_TRAIN:]

#####################
# ADD YOUR CODE BELOW
#####################
lambda_vals = [-1.5, -1.25, -1, -0.1, 0.01, 0.1, 1, 10, 100]

print("Train error rates for Fisher's LDA")
for lambda_val in lambda_vals:
    # instantiate LDA object     
    model = MyLDA(lambda_val)

    # call to your CV function to compute error rates for each fold
    err = my_cross_val(model, 'err_rates', X_train, y_train)

    # print error rates from CV
    print("Lambda: ", lambda_val, ": ", err, " | Average: ", np.mean(err), "S/D:", np.std(err))

# instantiate LDA object for best value of lambda
model = MyLDA(-1.25)

# fit model using all training data
model.fit(X_train, y_train)

# predict on test data
pred = model.predict(X_test)

# compute error rate on test data
errs = err_rate(pred, y_test)

# print error rate on test data
print(errs)
