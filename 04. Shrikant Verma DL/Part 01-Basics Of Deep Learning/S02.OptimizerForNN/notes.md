
### **Binary-case**

\( y_i \in \{0, 1\} \)

\[
\begin{cases} 
\omega^T z_i + b = z_i \\ 
\text{Sigmoid}(z_i) = P(y_i=1 | x_i, \omega, b) 
\end{cases}
\]

\[
\left( \frac{1}{1 + e^{-z_i}} = \frac{e^{z_i}}{1 + e^{z_i}} \right)
\]

**Log-loss: (Binary CE)**

\[
= -\left[ y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \right]
\]

--- 


### **Multi-class** \((3^k \text{-class})\)   
\(1, 2, 3\)

\[
\begin{align*}
\text{input} & \\
x_i \rightarrow & L_1, L_2, \ldots \\
\\
0 \rightarrow z_1 \rightarrow p_1 = P(y_i=1 \mid x_i, \theta) \\
0 \rightarrow z_2 \rightarrow p_2 = P(y_i=2 \mid x_i, \theta) \\
0 \rightarrow z_3 \rightarrow p_3 = P(y_i=3 \mid x_i, \theta) \\
\\
\text{output} & \\
0 \leq p_i \leq 1 \quad \text{and} \quad \sum_{i=1}^{3} p_i = 1 \\
\end{align*}
\]

---

### **Softmax:**  

\[
p_1 = \frac{e^{z_1}}{\sum_{i=1}^{k} e^{z_i}} \quad \quad 0 \leq p_i \leq 1
\]

\[
p_2 = \frac{e^{z_2}}{\sum_{i=1}^{k} e^{z_i}}
\]

\[
p_3 = \frac{e^{z_3}}{\sum_{i=1}^{k} e^{z_i}}
\]

if \( k = 3; \)

\[
p_1 + p_2 + p_3 = 1
\]

---

**softmax** \( \approx \) **sigmoid-like function for multi-class classification**

---
**(Q)** why \( e^{z} \) is common  

\[
\sqrt{\frac{\partial e^{z}}{\partial z}} = e^{z}
\]

---

