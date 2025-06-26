SciPy basics guide with code examples that are useful in NLP and general scientific computing.


---

🔹 1. Importing SciPy and Basic Submodules

import scipy
from scipy import stats, optimize, integrate, linalg, fft, spatial
import numpy as np


---

🔹 2. Statistical Functions (scipy.stats)

Probability Distribution - Normal (Gaussian)

from scipy.stats import norm

# PDF and CDF of standard normal
x = np.linspace(-5, 5, 100)
pdf = norm.pdf(x)
cdf = norm.cdf(x)

T-test (for NLP sentiment analysis or A/B testing)

from scipy.stats import ttest_ind

# Sample data
group1 = [23, 21, 19, 24, 30]
group2 = [27, 29, 30, 32, 35]

# Perform t-test
t_stat, p_val = ttest_ind(group1, group2)
print(f"T-Statistic: {t_stat}, P-Value: {p_val}")


---

🔹 3. Linear Algebra (scipy.linalg)

from scipy.linalg import inv, det

A = np.array([[1, 2], [3, 4]])

# Inverse and Determinant
A_inv = inv(A)
A_det = det(A)

print("Inverse:\n", A_inv)
print("Determinant:", A_det)


---

🔹 4. Optimization (scipy.optimize)

from scipy.optimize import minimize

# Function to minimize: f(x) = (x - 3)^2
f = lambda x: (x - 3)**2

# Minimize
result = minimize(f, x0=0)
print("Minimum at:", result.x)


---

🔹 5. Numerical Integration (scipy.integrate)

from scipy.integrate import quad

# Integrate f(x) = x^2 from 0 to 3
result, error = quad(lambda x: x**2, 0, 3)
print("Integral:", result)


---

🔹 6. Fourier Transform (scipy.fft)

from scipy.fft import fft

signal = [0, 1, 0, -1] * 4
transformed = fft(signal)
print("FFT:", transformed)


---

🔹 7. Distance Metrics (scipy.spatial.distance)

Useful in NLP for cosine similarity between sentence embeddings.

from scipy.spatial.distance import cosine

vec1 = np.array([1, 0, 1])
vec2 = np.array([0, 1, 1])

similarity = 1 - cosine(vec1, vec2)
print("Cosine Similarity:", similarity)


---

If you want examples specific to NLP (tokenizing, embeddings, etc.) using SciPy, let me know—I can tailor examples for that.

