import numpy as np
import matplotlib.pyplot as plt

X = np.array([155, 180, 164, 162, 181, 182, 173, 190, 171, 170, 181, 182, 189, 184, 209, 210])
Y = np.array([51, 52, 54, 53, 55, 59, 61, 59, 63, 76, 64, 66, 69, 72, 70, 80])

X_norm = (X - np.mean(X)) / np.std(X)

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

X_norm = np.column_stack((np.ones(len(X_norm)), X_norm))

theta = np.zeros(2)

alpha = 0.01
num_iters = 1500

theta, J_history = gradient_descent(X_norm, Y, theta, alpha, num_iters)

y_pred = hypothesis(X_norm, theta)
ss_tot = np.sum((Y - np.mean(Y))**2)
ss_res = np.sum((Y - y_pred)**2)
r_squared = 1 - (ss_res / ss_tot)

print(f"Hệ số chặn (b0): {theta[0]:.4f}")
print(f"Hệ số góc (b1): {theta[1]:.4f}")
print(f"R-squared: {r_squared:.4f}")

new_times = np.array([160, 170, 185])
new_times_norm = (new_times - np.mean(X)) / np.std(X)
new_times_norm = np.column_stack((np.ones(len(new_times_norm)), new_times_norm))

for i, time in enumerate(new_times):
    prediction = hypothesis(new_times_norm[i], theta)
    print(f"Điểm thi dự đoán cho thời gian học {time} phút: {prediction:.2f}")

plt.figure(figsize=(12, 8))

plt.scatter(X, Y, color='blue', label='Dữ liệu thực tế')

X_plot = np.linspace(min(X), max(X), 100)
X_plot_norm = (X_plot - np.mean(X)) / np.std(X)
X_plot_norm = np.column_stack((np.ones(len(X_plot_norm)), X_plot_norm))
y_plot = hypothesis(X_plot_norm, theta)
plt.plot(X_plot, y_plot, color='red', label='Đường hồi quy')

plt.xlabel('Thời gian học (phút)')
plt.ylabel('Điểm thi')
plt.title('Linear Regression: Thời gian học vs Điểm thi')
plt.legend()
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.plot(J_history)
plt.xlabel('Số lần lặp')
plt.ylabel('Giá trị hàm chi phí')
plt.title('Hàm chi phí qua các lần lặp')
plt.grid(True)

plt.show()