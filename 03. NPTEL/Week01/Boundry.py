import numpy as np  
import matplotlib.pyplot as plt  

# Data points  
class_1 = [(2, 4), (2, 0), (-1, 0), (-2, -2)]  
class_neg_1 = [(3, -1), (5, 6)]  

# Separate the points for plotting  
x1_class_1, y1_class_1 = zip(*class_1)  
x1_class_neg_1, y1_class_neg_1 = zip(*class_neg_1)  

# Create a scatter plot  
plt.figure(figsize=(8, 6))  

# Plot class 1 points  
plt.scatter(x1_class_1, y1_class_1, color='blue', label='Class 1 (y=1)', marker='o', s=100)  

# Plot class -1 points  
plt.scatter(x1_class_neg_1, y1_class_neg_1, color='red', label='Class -1 (y=-1)', marker='x', s=100)  

# Adding labels and title  
plt.title('Scatter Plot of Data Points with Decision Boundary')  
plt.xlabel('x1')  
plt.ylabel('x2')  

# Adding grid  
plt.grid(True)  

# Adding legend  
plt.legend()  

# Set axis limits for better visualization  
plt.xlim(-3, 6)  
plt.ylim(-3, 7)  

# Adding decision boundary  
# Example equation of a line: y = mx + c; let's use a slope (m) and intercept (c)  
# For this example, we'll use arbitrary slope and intercept for visualization  
# This is just a representative line and does not guarantee linear separability  

m = -1  # Slope  
c = 2   # y-intercept  

# Generate x values  
x_values = np.linspace(-3, 6, 100)  
# Calculate corresponding y values  
y_values = m * x_values + c  

# Plotting the decision boundary  
plt.plot(x_values, y_values, color='green', label='Decision Boundary', linestyle='--')  

# Show the plot  
plt.axhline(0, color='black',linewidth=0.5, ls='--')  
plt.axvline(0, color='black',linewidth=0.5, ls='--')  
plt.legend()  
plt.show()