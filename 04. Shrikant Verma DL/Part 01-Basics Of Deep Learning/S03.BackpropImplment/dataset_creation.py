import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to create a spiral dataset with an extra feature X3
def create_spiral_dataset(points_per_class, num_classes):
    X1, X2, X3, y = [], [], [], []
    for class_number in range(num_classes):
        ix = range(points_per_class * class_number, points_per_class * (class_number + 1))
        r = np.linspace(0.0, 1, points_per_class)  # Radius
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points_per_class) + np.random.randn(points_per_class) * 0.2  # Theta
        X1.extend(r * np.sin(t))  # X1 based on spiral
        X2.extend(r * np.cos(t))  # X2 based on spiral
        X3.extend(r + np.random.randn(points_per_class) * 0.1)  # X3 random feature with noise
        y.extend([class_number] * points_per_class)  # Class labels

    return np.array(X1), np.array(X2), np.array(X3), np.array(y)

# Parameters
points_per_class = 100
num_classes = 3

# Generate the dataset
X1, X2, X3, y = create_spiral_dataset(points_per_class, num_classes)

# Create a pandas DataFrame
df = pd.DataFrame({
    'X1': X1,
    'X2': X2,
    'X3': X3,
    'y': y
})

df.to_csv('multiclass_spiral_dataset.csv', index=False)

# Scatter plot for X1 vs X2 colored by y
plt.scatter(df['X1'], df['X2'], c=df['y'], cmap=plt.cm.Spectral, s=40)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Scatter plot of X1 vs X2 colored by class label y')
plt.colorbar(label='Class Label')
plt.show()
