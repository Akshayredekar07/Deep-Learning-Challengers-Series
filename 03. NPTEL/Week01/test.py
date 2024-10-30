import numpy as np  
import matplotlib.pyplot as plt  

# Data Points  
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  
y = np.array([1, 1, 1, 0])  # Labels  

# Plot data points  
plt.figure(figsize=(8, 6))  
for i, label in enumerate(y):  
    if label == 1:  
        plt.scatter(X[i, 0], X[i, 1], color='blue', label='Class 1' if i == 0 else "")  
    else:  
        plt.scatter(X[i, 0], X[i, 1], color='red', label='Class 0' if i == 3 else "")  

# Set x and y limits  
plt.xlim(-0.5, 1.5)  
plt.ylim(-0.5, 1.5)  

# Labels and title  
plt.xlabel('x1')  
plt.ylabel('x2')  
plt.title('Perceptron Learning in Linearly Separable Data')  
plt.axhline(0, color='black',linewidth=0.5, ls='--')  
plt.axvline(0, color='black',linewidth=0.5, ls='--')  

# Decision boundary: w1*x1 + w2*x2 + b = 0  
# For example, assume weights w = [1, 1] and bias b = -1.5  
# This gives the decision boundary x2 = -1*x1 + 1.5  
x_values = np.linspace(-0.5, 1.5, 100)  
y_values = -x_values + 1.5  
plt.plot(x_values, y_values, color='green', label='Decision Boundary')  

# Add legend  
plt.legend()  

# Show plot  
plt.grid()  
plt.show()