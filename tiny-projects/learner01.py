from sklearn.linear_model import LinearRegression
import numpy as np


# Training Data
x = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]) #Features
y = np.array([3, 7, 11, 15, 19, 23])  #Targets

# Define the learner algorithm
learner = LinearRegression()

#train the learner (fit the model)
learner.fit(x, y)

# Output the optimal parameters from hypothesis space
print("Learned weight:", learner.coef_)
print("Learned bias(intercept):", learner.intercept_)


# Make predictions
print("Predictions for nx [14,15]", learner.predict([[14, 15]]))