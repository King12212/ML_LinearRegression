import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('Practice2_Chapter2.csv')

X = np.array([data['TV'], data['Radio'], data['Newspaper']]).T
y = np.array(data['Sales'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def hypothesis(X, theta):
    return np.dot(X, theta)

def cost_function(X, y, theta):
    m = len(y)
    predictions = hypothesis(X, theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    
    for _ in range(num_iters):
        predictions = hypothesis(X, theta)
        theta = theta - (alpha / m) * np.dot(X.T, (predictions - y))
        J_history.append(cost_function(X, y, theta))
    
    return theta, J_history

X_train_scaled = np.column_stack((np.ones(len(X_train_scaled)), X_train_scaled))
X_test_scaled = np.column_stack((np.ones(len(X_test_scaled)), X_test_scaled))

theta = np.zeros(X_train_scaled.shape[1])

alpha = 0.01
num_iters = 1500

theta, J_history = gradient_descent(X_train_scaled, y_train, theta, alpha, num_iters)

y_pred = hypothesis(X_test_scaled, theta)
ss_tot = np.sum((y_test - np.mean(y_test))**2)
ss_res = np.sum((y_test - y_pred)**2)
r_squared = 1 - (ss_res / ss_tot)

print("Hệ số:")
print(f"Hệ số chặn (b0): {theta[0]:.4f}")
print(f"TV (b1): {theta[1]:.4f}")
print(f"Radio (b2): {theta[2]:.4f}")
print(f"Newspaper (b3): {theta[3]:.4f}")
print(f"R-squared trên tập kiểm tra: {r_squared:.4f}")

new_data = np.array([[200, 50, 100]])  # TV: 200, Radio: 50, Newspaper: 100
new_data_scaled = scaler.transform(new_data)
new_data_scaled = np.column_stack((np.ones(len(new_data_scaled)), new_data_scaled))
prediction = hypothesis(new_data_scaled, theta)
print(f"Doanh số dự đoán cho ngân sách quảng cáo (TV: $200k, Radio: $50k, Newspaper: $100k): ${prediction[0]:.2f}k")

plt.figure(figsize=(10, 6))
plt.plot(J_history)
plt.xlabel('Số lần lặp')
plt.ylabel('Giá trị hàm chi phí')
plt.title('Hàm chi phí qua các lần lặp')
plt.grid(True)

plt.show()

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_test[:, 0], X_test[:, 1], y_test, c='blue', marker='o', alpha=0.5)
ax.set_xlabel('TV Advertising')
ax.set_ylabel('Radio Advertising')
ax.set_zlabel('Sales')
ax.set_title('3D Scatter plot: TV & Radio Advertising vs Sales')

plt.show()