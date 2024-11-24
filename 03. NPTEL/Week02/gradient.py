import numpy as np  
X = [0.5, 2.5]  # Features (input values)
Y = [0.2, 0.9]  # Targets (expected output values)

# Sigmoid function: computes the probability output for logistic regression
def f(w, b, x):
    # The sigmoid function formula: 1 / (1 + exp(-(w * x + b)))
    return 1.0 / (1.0 + np.exp(-(w * x + b)))

# Error function: calculates the total Mean Squared Error (MSE) over all data points
def error(w, b):
    err = 0.0  # Initialize the total error to 0
    for x, y in zip(X, Y):  # Loop through all input-output pairs
        fx = f(w, b, x)  # Compute the model's prediction using the sigmoid function
        err += 0.5 * (fx - y) ** 2  # Add the squared difference (scaled by 0.5) to the error
    return err  # Return the total error

# Gradient of the error with respect to the bias (b)
def grad_b(w, b, x, y):
    fx = f(w, b, x)  # Compute the model's prediction using the sigmoid function
    # Gradient formula: (fx - y) * fx * (1 - fx)
    return (fx - y) * fx * (1 - fx)

# Gradient of the error with respect to the weight (w)
def grad_w(w, b, x, y):
    fx = f(w, b, x)  # Compute the model's prediction using the sigmoid function
    # Gradient formula: (fx - y) * fx * (1 - fx) * x
    return (fx - y) * fx * (1 - fx) * x

# Perform Gradient Descent: update weights (w) and bias (b) to minimize error
def do_gradient_descent():
    # Initialize the weight (w) and bias (b) with arbitrary starting values
    w, b = -2, -2
    
    # Set the learning rate (eta) and the maximum number of iterations (epochs)
    eta = 1.0  # Learning rate controls how big the updates are
    max_epochs = 1000  # Number of iterations to update w and b
    
    for i in range(max_epochs):  # Iterate for the given number of epochs
        dw, db = 0, 0  # Initialize gradients for w and b to 0
        
        # Loop through all data points to calculate the gradients
        for x, y in zip(X, Y):  # x: feature, y: target value
            dw += grad_w(w, b, x, y)  # Accumulate the gradient for w
            db += grad_b(w, b, x, y)  # Accumulate the gradient for b
        
        # Update the weight and bias using the gradients and learning rate
        w -= eta * dw
        b -= eta * db

        # Every 100 epochs, print the current error to monitor progress
        if i % 100 == 0:
            print(f"Epoch {i}: Error = {error(w, b):.4f}")
    
    # Return the final values of weight (w) and bias (b) after optimization
    return w, b

# Run the gradient descent algorithm
final_w, final_b = do_gradient_descent()

# Print the final optimized values of w and b
print(f"Final weights: w = {final_w}, b = {final_b}")

