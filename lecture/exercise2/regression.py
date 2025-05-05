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

iris = load_iris()

# Create a DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
print(df.head(10))


decisionTreeRegressor_model = DecisionTreeRegressor()
decisionTreeRegressor_params = decisionTreeRegressor_model.get_params()
print(decisionTreeRegressor_params)

