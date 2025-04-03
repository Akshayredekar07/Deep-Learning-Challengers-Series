## Training an MLP

An **MLP** is like a brain with layers of “neurons” that learns to guess answers from data (e.g., predicting house prices). Let’s break it down simply with a full example.

### Example Setup
- **Input**: $x = [2, 3]$ (2 features, like house size and rooms).  
- **Output**: $y = 10$ (actual price we want to predict).  
- **Layers**:  
  - Input: 2 neurons.  
  - Hidden: 2 neurons.  
  - Output: 1 neuron.  
- **Learning Rate**: $\eta = 0.1$ (how fast we adjust).

### 1. Start with Random Weights and Biases
- **Weights**: Numbers that connect neurons.  
  - From input to hidden:  
    $$
    W1 = \begin{bmatrix} 0.5 & 0.2 \\ 0.3 & 0.4 \end{bmatrix}
    $$  
  - From hidden to output:  
    $$
    W2 = \begin{bmatrix} 0.6 \\ 0.7 \end{bmatrix}
    $$  
- **Biases**: Extra tweaks for each neuron.  
  - Hidden layer: $b1 = [0.1, 0.1]$.  
  - Output layer: $b2 = 0.1$.

### 2. Forward Pass (Make a Guess)
We push $x = [2, 3]$ through the network to get a prediction.

#### Step 1: Input to Hidden Layer
- Calculate “signal” for hidden neurons ($z1$):  
  $$
  z1 = W1 \cdot x + b1
  $$  
  - First neuron:  
    $$
    z1_1 = (0.5 \times 2) + (0.2 \times 3) + 0.1 = 1 + 0.6 + 0.1 = 1.7
    $$  
  - Second neuron:  
    $$
    z1_2 = (0.3 \times 2) + (0.4 \times 3) + 0.1 = 0.6 + 1.2 + 0.1 = 1.9
    $$  
  - So, $z1 = [1.7, 1.9]$.  

- Apply an “activation” (here we use ReLU: if $z > 0$, keep it; else, 0):  
  $$
  h = \text{ReLU}(z1) = [1.7, 1.9]
  $$ (both positive, so no change).

#### Step 2: Hidden to Output Layer
- Calculate output signal ($z2$):  
  $$
  z2 = W2 \cdot h + b2
  $$  
  - $$
    z2 = (0.6 \times 1.7) + (0.7 \times 1.9) + 0.1 = 1.02 + 1.33 + 0.1 = 2.45
    $$  
- Prediction: $\hat{y} = z2 = 2.45$.

### 3. Backward Pass (Adjust Weights)
We define the loss as $L = \frac{1}{2}(y - \hat{y})^2$ (half squared error for simplicity). Now, tweak weights and biases to lower the error by finding “slopes” (gradients) and adjusting backward.

#### Step 1: Output Layer Gradient
- How much does $L$ change with $\hat{y}$?  
  $$
  \frac{dL}{d\hat{y}} = -(y - \hat{y}) = -(10 - 2.45) = -7.55
  $$  
- Since $\hat{y} = z2$ (no activation), this is also $\frac{dL}{dz2}$.

- Gradient for $W2$:  
  $$
  \frac{dL}{dW2} = \frac{dL}{dz2} \cdot h
  $$  
  - $dW2_1 = -7.55 \times 1.7 = -12.835$  
  - $dW2_2 = -7.55 \times 1.9 = -14.345$  

- Gradient for $b2$:  
  $$
  \frac{dL}{db2} = \frac{dL}{dz2} = -7.55
  $$

- Update $W2$ and $b2$:  
  $$
  W2_{\text{new}} = W2 - \eta \cdot \frac{dL}{dW2}
  $$  
  - $W2_1 = 0.6 - 0.1 \times (-12.835) = 0.6 + 1.2835 = 1.8835$  
  - $W2_2 = 0.7 - 0.1 \times (-14.345) = 0.7 + 1.4345 = 2.1345$  
  - New $W2 = [1.8835, 2.1345]$.  
  $$
  b2_{\text{new}} = 0.1 - 0.1 \times (-7.55) = 0.1 + 0.755 = 0.855
  $$

#### Step 2: Hidden Layer Gradient
- Error at hidden layer ($dh$):  
  $$
  dh = \frac{dL}{dz2} \cdot W2
  $$  
  - $dh_1 = -7.55 \times 0.6 = -4.53$  
  - $dh_2 = -7.55 \times 0.7 = -5.285$  
  - $dh = [-4.53, -5.285]$.  

- Since ReLU’s derivative is 1 when $z > 0$ (and $z1 = [1.7, 1.9]$ are positive):  
  $$
  dz1 = dh = [-4.53, -5.285]
  $$

- Gradient for $W1$:  
  $$
  \frac{dL}{dW1} = dz1 \cdot x
  $$  
  - $dW1_{11} = -4.53 \times 2 = -9.06$  
  - $dW1_{12} = -4.53 \times 3 = -13.59$  
  - $dW1_{21} = -5.285 \times 2 = -10.57$  
  - $dW1_{22} = -5.285 \times 3 = -15.855$

- Update $W1$:  
  $$
  W1_{\text{new}} = W1 - \eta \cdot \frac{dL}{dW1}
  $$  
  - $W1_{11} = 0.5 - 0.1 \times (-9.06) = 0.5 + 0.906 = 1.406$  
  - $W1_{12} = 0.2 - 0.1 \times (-13.59) = 0.2 + 1.359 = 1.559$  
  - $W1_{21} = 0.3 - 0.1 \times (-10.57) = 0.3 + 1.057 = 1.357$  
  - $W1_{22} = 0.4 - 0.1 \times (-15.855) = 0.4 + 1.5855 = 1.9855$  
  - New $W1 = \begin{bmatrix} 1.406 & 1.559 \\ 1.357 & 1.9855 \end{bmatrix}$.

- Gradient for $b1$:  
  $$
  \frac{dL}{db1} = dz1 = [-4.53, -5.285]
  $$  
- Update $b1$:  
  - $b1_1 = 0.1 - 0.1 \times (-4.53) = 0.1 + 0.453 = 0.553$  
  - $b1_2 = 0.1 - 0.1 \times (-5.285) = 0.1 + 0.5285 = 0.6285$  
  - New $b1 = [0.553, 0.6285]$.

### 4. Repeat Until Good Enough
- Run steps 2–3 again with new $W1$, $W2$, $b1$, $b2$.  
- Each time, $\hat{y}$ gets closer to $y = 10$, and $L$ shrinks.  
- Stop when $L$ is tiny (e.g., after many rounds).