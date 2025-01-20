
Binary Classification: 
$y_i \in \{0, 1\}$

Logistic Regression:
$z_i = w^Tx_i + b$

$p_i = \text{Sigmoid}(z_i) = P(y_i=1 | x_i, w, b)$

$\text{Sigmoid}(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}$
$P(y_i=0 | x_i, w, b) = 1 - p_i$

---

**Multi-class Classification**

*   **Number of Classes:** K = 3 (In this example, there are three classes: 1, 2, and 3)
*   **Input:** xᵢ (Input features)
*   **Network Parameters:** θ (All parameters of the Neural Network)

**Process**

1.  **Input to Outputs:** The input xᵢ is processed through the network, resulting in three outputs: z₁, z₂, and z₃.

2.  **Probabilities:** Each output zᵢ corresponds to the probability of the input belonging to a specific class:
    *   P₁ = P(yᵢ = 1 | xᵢ, θ) (Probability that yᵢ belongs to class 1, given xᵢ and θ)
    *   P₂ = P(yᵢ = 2 | xᵢ, θ) (Probability that yᵢ belongs to class 2, given xᵢ and θ)
    *   P₃ = P(yᵢ = 3 | xᵢ, θ) (Probability that yᵢ belongs to class 3, given xᵢ and θ)

**Constraints**

*   **Probability Range:** 0 ≤ Pᵢ ≤ 1 (Each probability must be between 0 and 1, inclusive).
*   **Sum of Probabilities:** Σ (from i=1 to K) Pᵢ = 1 (The sum of all probabilities must equal 1).

**Softmax Function**

The softmax function is used in multi-class classification to convert a vector of raw scores (zᵢ) into a probability distribution.

**Formula**

For a K-class classification problem, the probability Pᵢ for class i is calculated as follows:

$P_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$

Where:

*   $P_i$: Probability of belonging to class i.
*   $z_i$: Raw score (output of the neural network) for class i.
*   $K$: Total number of classes.

**Example (K=3)**

If we have 3 classes (K=3), the probabilities are calculated as:

$P_1 = \frac{e^{z_1}}{e^{z_1} + e^{z_2} + e^{z_3}}$

$P_2 = \frac{e^{z_2}}{e^{z_1} + e^{z_2} + e^{z_3}}$

$P_3 = \frac{e^{z_3}}{e^{z_1} + e^{z_2} + e^{z_3}}$

**Properties**

*   $0 \le P_i \le 1$ (Each probability is between 0 and 1, inclusive).
*   $\sum_{i=1}^{K} P_i = 1$ (The sum of all probabilities equals 1).

In the example with K=3:

$P_1 + P_2 + P_3 = 1$
