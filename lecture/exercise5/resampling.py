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


X_data = df_california[["MedInc"]]
y_data = df_california[["MedHouseVal"]]

test_errors = []

for share in np.arange(0.1, 1.0, 0.1):
    errors = []
    for _ in range(10):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, train_size=share, random_state=42
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        errors.append(mse)
    test_errors.append(errors)

print("Test Errors for different training set sizes:")
for share, errors in zip(np.arange(0.1, 1.0, 0.1), test_errors):
    print(f"Training set size {int(share * 100)}%: Mean MSE = {np.mean(errors):.3f}, Std MSE = {np.std(errors):.3f}")




training_shares = np.arange(0.1, 1.0, 0.1)

plt.figure(figsize=(8, 5))

plt.boxplot(test_errors, positions=training_shares, widths=0.05, patch_artist=True)

plt.xlabel('Training Data Share')
plt.ylabel('Test Mean Squared Error (MSE)')
plt.title('Test Error vs Training Data Size')

plt.xticks(training_shares, [f'{int(x*100)}%' for x in training_shares])
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.show()