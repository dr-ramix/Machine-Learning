import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_california_housing

dataset_california = fetch_california_housing(as_frame=True)

df_california =dataset_california.frame.loc[:, ["MedInc", "MedHouseVal"]]

print("California Housing Dataset:")
print(df_california.head())


california_housing_figure = plt.figure(figsize=(4, 4))

ax = california_housing_figure.add_axes([0, 0, 1, 1])

sp = ax.scatter(
    x = df_california["MedInc"],
    y = df_california["MedHouseVal"],
)


print("length of the dataset:", len(df_california))


training_set = df_california.iloc[:100]
test_set = df_california.iloc[100:]



def evaluate_with_mse(model, train_set, test_set) -> str:

    x_train = training_set[["MedInc"]]
    y_train = training_set[["MedHouseVal"]]
    x_test = test_set[["MedInc"]]
    y_test = test_set[["MedHouseVal"]]

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    result = f"MSE on Test-Data: {mse:.3f}."
    return result


evaluation_result = evaluate_with_mse(LinearRegression(), training_set, test_set)

print(evaluation_result)