# General libraries
import numpy as np
import pandas as pd

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn libraries
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

#load the iris dataset from sklearn.datasets module
iris = load_iris()

# Create a DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

#print the first 10 rows of the DataFrame
print(df.head(10))

# Creating a hypothesis space for Decision Tree Regressor
decisionTreeRegressor_model = DecisionTreeRegressor()

#Getting parameters of Decision Tree Regressor
decisionTreeRegressor_params = decisionTreeRegressor_model.get_params()
print(decisionTreeRegressor_params)

#splitting the dataset into features and target variable
features = iris.data
output = iris.target

# Split data into train and test sets
feature_train, feature_test, output_train, output_test = train_test_split(features, output, test_size=0.2, random_state=42)

# Create and train the DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=3, random_state=42)
model.fit(feature_train, output_train)

# Make predictions
output_pred = model.predict(feature_test)

#Evaluate the model
mse = mean_absolute_error(output_test, output_pred)
print("Mean absolute Error:", mse)

# Print model parameters
print("Model Parameters:\n", model.get_params())