import numpy as np

def my_cross_val(model, loss_func, X, y, k=10):
    err_rates = []

    train = np.array_split(X, k)
    labels = np.array_split(y, k)

    for i in range(k):
        x_ = train.copy()
        y_ = labels.copy()

        del x_[i]
        del y_[i]

        x_ = np.concatenate(x_)
        y_ = np.concatenate(y_)

        model.fit(x_, y_)
        predictions = model.predict(train[i])

        if loss_func == 'mse':
            err_rates.append(mse(predictions, labels[i]))
        else:
            err_rates.append(err_rate(predictions, labels[i]))

    return err_rates

def mse(yhat, y):
    return np.square(np.subtract(y, yhat)).mean()

def err_rate(yhat, y):
    return np.mean([yhat != y])