import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler

# Load data
house_data = pd.read_csv("dataSet/train.csv")
df = pd.DataFrame(house_data)
y_train = df["PRICE"].to_numpy().reshape(-1, 1)
x_train = df["PROPERTYSQFT"].to_numpy().reshape(-1, 1)

# Scale data
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_scaled = scaler_x.fit_transform(x_train)
y_scaled = scaler_y.fit_transform(y_train)

# Plot the scaled data
plt.scatter(x_scaled, y_scaled, marker="x", c="r")
plt.title("House Price to Area Summary")
plt.xlabel("House Area in FT")
plt.ylabel("House Price in Lakhs")
plt.show()

def compute_cost(x, y, w, b):
    m = x.shape[0]
    fw_b = w * x + b
    error = fw_b - y
    total = np.sum(error ** 2)
    cost_fn = (1 / (2 * m)) * total
    return cost_fn

initial_w = 2
initial_b = 1
cost = compute_cost(x_scaled, y_scaled, initial_w, initial_b)
print(f'Cost at initial w: {cost}')

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dJ_dw = (1/m) * np.sum((w * x + b - y) * x)
    dJ_db = (1/m) * np.sum(w * x + b - y)
    return dJ_dw, dJ_db

def gradient_descent(x, y, w, b, cost_func, gradient_func, alpha, iteration):
    for i in range(iteration):
        dJ_dw, dJ_db = gradient_func(x, y, w, b)
        w -= alpha * dJ_dw
        b -= alpha * dJ_db

        if i % math.ceil(iteration/10) == 0:
            cost = cost_func(x, y, w, b)
            print(f"Iteration {i:4}: Cost {cost}")

    return w, b

# Gradient descent parameters
initial_w = 1
initial_b = 1
alpha = 0.001
iteration = 10000
w, b = gradient_descent(x_scaled, y_scaled, initial_w, initial_b, compute_cost, compute_gradient, alpha, iteration)

print(f"Optimized weight: {w}")
print(f"Optimized bias: {b}")

# Prediction
model_predict = float(input("Enter house area here: "))
scaled_input = scaler_x.transform(np.array([[model_predict]]))
predicted_scaled_price = w * scaled_input + b
predicted_price = scaler_y.inverse_transform(predicted_scaled_price)

print(f"Your Predicted House Price is: {predicted_price[0][0]}")
