# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries.
2.Initialize feature matrix X and target vector y.
3.Create the SGD Regressor with specified parameters.
4.Train the model using fit().
5.Obtain model coefficients and intercept. 
6.Predict output values using predict(). 
7.Plot actual versus predicted values.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: A.VIMAL
RegisterNumber:  212224240183
*/
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = np.array([
    [800, 2],
    [1000, 3],
    [1200, 3],
    [1500, 4],
    [1800, 4],
    [2000, 5],
    [2200, 5],
    [2500, 6]
])

y = np.array([
    [40, 2],
    [55, 3],
    [65, 3],
    [85, 4],
    [95, 4],
    [110, 5],
    [125, 5],
    [145, 6]
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sgd = SGDRegressor(max_iter=2000, eta0=0.01, learning_rate='constant', random_state=42)
model = MultiOutputRegressor(sgd)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Predicted [Price, Occupants]:")
print(y_pred)

print("\nActual [Price, Occupants]:")
print(y_test)

mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:", round(mse, 2))

new_house = np.array([[1800, 8]])
new_house_scaled = scaler.transform(new_house)
new_prediction = model.predict(new_house_scaled)

print("\nFor New House [1600 sq ft, 4 rooms]:")
print("Predicted House Price (lakhs):", round(new_prediction[0][0], 2))
print("Predicted Number of Occupants:", round(new_prediction[0][1]))

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(X_test[:, 0], y_test[:, 0], color='green', label="Actual Price")
plt.scatter(X_test[:, 0], y_pred[:, 0], color='red', marker='x', label="Predicted Price")
plt.xlabel("House Size (sq ft)")
plt.ylabel("House Price (lakhs)")
plt.title("House Size vs House Price")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.scatter(X_test[:, 0], y_test[:, 1], color='blue', label="Actual Occupants")
plt.scatter(X_test[:, 0], y_pred[:, 1], color='orange', marker='x', label="Predicted Occupants")
plt.xlabel("House Size (sq ft)")
plt.ylabel("Number of Occupants")
plt.title("House Size vs Number of Occupants")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

```

## Output:


<img width="1067" height="695" alt="Screenshot 2026-05-12 073215" src="https://github.com/user-attachments/assets/bdb86ca8-3dfb-4ded-84a6-5dc3bd6e98ff" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
