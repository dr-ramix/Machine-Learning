import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 1. Generate synthetic non-linear data
np.random.seed(1)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])  # Add some noise

# 2. Define degrees of the polynomial to fit
degrees = [1, 3, 9]

# 3. Plot the results
plt.figure(figsize=(18, 5))

for i, degree in enumerate(degrees, 1):
    # Create polynomial features and fit the model
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    
    # Generate predictions
    X_test = np.linspace(0, 5, 100).reshape(-1, 1)
    y_pred = model.predict(X_test)
    
    # Plot
    plt.subplot(1, 3, i)
    plt.scatter(X, y, color='gray', label='Data')
    plt.plot(X_test, y_pred, color='blue', label=f'Degree {degree}')
    plt.title(f'Polynomial Degree {degree}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

plt.tight_layout()
plt.show()
