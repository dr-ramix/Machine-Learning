import numpy as np
import pandas as pd

import math

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


#Creating a synthetic dataset

#TRAINING DATA
np.random.seed(43)
# Generate Feature date
x_train = np.linspace(10, 15, num=50)
# Generate Target data with sinusoidal function
y_train = 10 + 3 * np.sin(0.15 * math.pi * x_train)
# Adding some Gaussian noise to the target variable
y_train += np.random.normal(loc=0.0, scale=0.5, size=len(x_train))
# Create a DataFrame for training data
data_train = pd.DataFrame({"y": y_train, "x": x_train})

#TESTING DATA
#For Testing, we will use a different seed to generate new data
np.random.seed(2238)
x_test = np.linspace(10, 15, num=50)
y_test = 10 + 3 * np.sin(0.15 * math.pi * x_test)
y_test += np.random.normal(loc=0.0, scale=0.5, size=len(x_test))
data_test = pd.DataFrame({"y": y_test, "x": x_test})


def evaluate_model_with_mse(model, train_set, test_set) -> str:

    x_train = train_set[['x']]
    y_train = train_set[['y']]
    x_test = test_set[['x']]
    y_test = test_set[['y']]

    # Fit the model
    model.fit(x_train, y_train)

    # Predict on training and testing sets
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Calculate MSE for training and testing sets
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    result = f"MSE on Train-Data: {mse_train:.3f} and MSE on Test-Data {mse_test:.3f}."

    return result


def evaluate_model_with_mae(model, train_set, test_set) -> str:

    x_train = train_set[['x']]
    y_train = train_set[['y']]
    x_test = test_set[['x']]
    y_test = test_set[['y']]

    # Fit the model
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    test_pred  = model.predict(x_test)

    # Calculate MAE for training and testing sets
    mae_train = mean_absolute_error(y_train, train_pred)
    mae_test  = mean_absolute_error(y_test, test_pred)

    result = f"MAE on Train-Data: {mae_train:.3f} and MAE on Test-Data {mae_test:.3f}."

    return result

evaluation_with_mse = evaluate_model_with_mae(LinearRegression(), data_train, data_test)
print(evaluation_with_mse)

evaluation_with_mae = evaluate_model_with_mse(LinearRegression(), data_train, data_test)
print(evaluation_with_mae)
