### 1. **Vocabulary and One-Hot Encoding Definition**

   - **Vocabulary** $V$: The set of all unique words in a corpus (e.g., all unique words in a document or collection of sentences).  
   - **Vocabulary Size** $|V|$: The total number of unique words in $V$.
   - **One-Hot Representation**: A vector representation for each word in the vocabulary where only one entry is "1" (indicating the word's position in $V$) and all other entries are "0".

#### Example:
   If $V = \{\text{"cat"}, \text{"dog"}, \text{"truck"}\}$, we can represent each word as:
   - **"cat"**: $([1, 0, 0])$
   - **"dog"**: $([0, 1, 0])$
   - **"truck"**: $([0, 0, 1])$

   If $|V| = 3$, then each one-hot vector has a length of 3.

### 2. **Mathematical Form of One-Hot Encoding**

   - For a vocabulary $V$ with size $|V|$, each word $w$ in $V$ is represented by a vector $x_w \in \mathbb{R}^{|V|}$.
   - In $x_w$, only one element is "1" (the position corresponding to $w$) and all other elements are "0".

   #### Example:
   If $V = \{\text{"human"}, \text{"machine"}, \text{"interface"}\}$, then:
   - "human" = $([1, 0, 0])$
   - "machine" = $([0, 1, 0])$
   - "interface" = $([0, 0, 1])$

### 3. **Limitations of One-Hot Encoding**

   - **High Dimensionality**: 
      - For large vocabularies, $|V|$ can be very large (e.g., $50{,}000$ words in the Penn Treebank, $13$ million in Google's corpus).
      - Each word requires a vector of length $|V|$, which becomes computationally expensive and memory-intensive.

   - **No Semantic Similarity**:
      - In one-hot encoding, the Euclidean distance and cosine similarity between any two distinct words are constant, regardless of semantic meaning.
      - This implies that "cat" and "dog" (similar concepts) are as dissimilar as "cat" and "truck" (different concepts).

   #### Example Calculations:
   - **Euclidean Distance** between two one-hot vectors $x_{cat}$ and $x_{dog}$:
     
     $\text{distance}(x_{cat}, x_{dog}) = \sqrt{\sum_{i=1}^{|V|} (x_{cat,i} - x_{dog,i})^2} = \sqrt{2}$
   
   - **Cosine Similarity** between $x_{cat}$ and $x_{dog}$:
     
     $\text{cosine\_similarity}(x_{cat}, x_{dog}) = \frac{x_{cat} \cdot x_{dog}}{\|x_{cat}\| \|x_{dog}\|} = 0$

### 4. **Need for Dense Embeddings**

   Since one-hot vectors don't capture the meaning or relationships between words, dense embeddings are used in natural language processing.

   - **Dense Embeddings**: These are learned, lower-dimensional vectors (e.g., word2vec, GloVe) that position semantically similar words closer together in vector space.
   - Unlike one-hot encoding, these embeddings capture word relationships and similarities, making them useful for tasks where the model needs to understand word meaning and context.

#### Example:
   If "cat" and "dog" are semantically close, a dense embedding might represent them as follows:
   - "cat": $([0.7, 0.5, 0.3])$
   - "dog": $([0.8, 0.4, 0.3])$

   In this space:
   - The **Euclidean Distance** and **Cosine Similarity** between "cat" and "dog" are smaller than the distance/similarity between "cat" and "truck."



### 1. **Distributed Representations of Words**
   - **Concept**: The idea is that the meaning of a word can be inferred from the words surrounding it. This concept was famously summarized by linguist J.R. Firth as "You shall know a word by the company it keeps."
   - **Objective**: Use the context (neighboring words) around a word to represent it, making it possible to capture semantic relationships between words in a vector space.

### 2. **Co-Occurrence Matrix**

   - **Definition**: A co-occurrence matrix is a table where each entry represents the number of times a particular word appears in the context of another word.
   - **Context Window**: A "window" of $k$ words around the target word (e.g., 2 words before and 2 words after) is typically used to define "context."
   - **Matrix Structure**: Rows and columns represent words. Each cell $(i, j)$ contains the count of times word $w_i$ appears in the context of word $w_j$.

#### Example
Given the following corpus:
   ```
   "Human machine interface for computer applications"
   "User opinion of computer system response time"
   "User interface management system"
   "System engineering for improved response time"
   ```

   **Vocabulary**: Suppose our vocabulary includes the words: `["human", "machine", "system", "for", "user"]`.

   **Co-occurrence Matrix with $k = 2$**:

   |        | human | machine | system | for   | user |
   |--------|-------|---------|--------|-------|------|
   | human  | 0     | 1       | 0      | 1     | 0    |
   | machine| 1     | 0       | 0      | 1     | 0    |
   | system | 0     | 0       | 0      | 1     | 2    |
   | for    | 1     | 1       | 1      | 0     | 0    |
   | user   | 0     | 0       | 2      | 0     | 0    |

Each row (or column) of this matrix serves as a vectorial representation of a word based on its co-occurrence with other words in the corpus.

### 3. **Issues with Raw Co-Occurrence Counts**
   - **Stop Words**: Common words (e.g., "a," "the," "for") appear very frequently, leading to high counts that do not contribute to meaningful representations.
   - **Sparseness**: Many pairs may not co-occur, leading to zeros in the matrix.
   - **High Counts Dominating Semantics**: Frequent co-occurrences do not necessarily imply meaningful relationships.

#### Solutions
   - **Solution 1**: Ignore very frequent words (stop words).
   - **Solution 2**: Use a threshold $t$ to cap counts. Define $X_{ij} = \min(\text{count}(w_i, c_j), t)$.

### 4. **Using Pointwise Mutual Information (PMI)**

   - **Definition**: PMI measures how much more often two words $w$ and $c$ co-occur than if they were independent. This helps in emphasizing meaningful co-occurrences.
   - **Formula**:
     $\text{PMI}(w, c) = \log \frac{p(w, c)}{p(w) \cdot p(c)} = \log \frac{\text{count}(w, c) \cdot N}{\text{count}(w) \cdot \text{count}(c)}$
     where:
     - $\text{count}(w, c)$: Number of times word $w$ and context $c$ co-occur.
     - $N$: Total number of word-context pairs in the corpus.

   - **Example Calculation**:
     Suppose:
     - $\text{count}(\text{machine}, \text{system}) = 2$
     - $N = 100$
     - $\text{count}(\text{machine}) = 10$
     - $\text{count}(\text{system}) = 5$

     Then:
     $\text{PMI}(\text{machine}, \text{system}) = \log \frac{2 \cdot 100}{10 \cdot 5} = \log \frac{200}{50} = \log 4 = 2$

   - **Adjusted PMI (PPMI)**:
     - To avoid negative PMI values, we can use Positive PMI (PPMI):
       $\text{PPMI}(w, c) = \max(\text{PMI}(w, c), 0)$
     - This transformation zeros out negative PMI values, retaining only positive associations.

### 5. **Final Co-Occurrence Matrix with PPMI**

   Applying PPMI can transform our matrix to reflect more meaningful associations:

   |        | human | machine | system | for   | user |
   |--------|-------|---------|--------|-------|------|
   | human  | 0     | 2.944   | 0      | 2.25  | 0    |
   | machine| 2.944 | 0       | 0      | 2.25  | 0    |
   | system | 0     | 0       | 0      | 1.15  | 1.84 |
   | for    | 2.25  | 2.25    | 1.15   | 0     | 0    |
   | user   | 0     | 0       | 1.84   | 0     | 0    |

This matrix, with PPMI values, now better represents the semantic relationships between words based on their contexts, as words that appear together in meaningful ways are emphasized.


### Problems with Co-Occurrence Matrices
Although co-occurrence matrices capture word meanings to an extent, they also face several significant challenges:

1. **High Dimensionality**: 
   - The matrix dimension is equal to the vocabulary size ($|V|$), which can be extremely large, especially with a substantial corpus.
   - **Example**: For a vocabulary of 50,000 words, the matrix would have 50,000 rows and columns, resulting in a $50,000 \times 50,000$ matrix.

2. **Sparsity**:
   - Co-occurrence matrices are sparse because each word only co-occurs with a small subset of other words.
   - **Result**: Most of the matrix cells contain zeroes, leading to memory inefficiency and computational challenges.

3. **Growth with Vocabulary Size**:
   - As the vocabulary size increases, the matrix dimension grows quadratically, making it increasingly challenging to store and process.

   **Example**:
   The example matrix from earlier illustrates this sparsity, with most values close to zero except for a few meaningful co-occurrence values:
   |         | human | machine | system | for   | user |
   |---------|-------|---------|--------|-------|------|
   | human   | 0     | 2.944   | 0      | 2.25  | 0    |
   | machine | 2.944 | 0       | 0      | 2.25  | 0    |
   | system  | 0     | 0       | 0      | 1.15  | 1.84 |
   | for     | 2.25  | 2.25    | 1.15   | 0     | 0    |
   | user    | 0     | 0       | 1.84   | 0     | 0    |

### Solution: Dimensionality Reduction Using Singular Value Decomposition (SVD)

To address these issues, **dimensionality reduction** techniques like Singular Value Decomposition (SVD) are commonly applied to co-occurrence matrices.

1. **What is SVD?**
   - SVD decomposes a matrix $X$ into three matrices:
     $X = U \Sigma V^T$
     - $U$: Matrix of the left singular vectors (word representations).
     - $\Sigma$: Diagonal matrix with singular values, which indicate the importance of each component.
     - $V$: Matrix of the right singular vectors (context representations).
   - By keeping only the top $k$ singular values in $\Sigma$, we reduce the dimensionality of $X$.

2. **Benefits of SVD for Word Representations**:
   - **Lower Dimensionality**: Reduces the matrix from $|V|$-dimensional to $k$-dimensional, where $k$ is much smaller (e.g., 100–300).
   - **Dense Representations**: Transforms sparse vectors into dense vectors, where each component has some meaningful information.
   - **Capturing Semantics**: Helps capture latent semantic relationships, as similar words are now closer in the reduced vector space.

3. **Example of Reduced Co-Occurrence Matrix**:
   - After applying SVD, each word now has a lower-dimensional representation, say $\mathbb{R}^{300}$ instead of $\mathbb{R}^{50,000}$, making it computationally efficient and capturing more meaningful relationships.

Using SVD to reduce the dimensionality of a co-occurrence matrix enables us to work with compact, dense, and semantically meaningful word representations.


### SVD for Learning Word Representations

The goal of **Singular Value Decomposition (SVD)** is to reduce a high-dimensional matrix to a lower-dimensional form while preserving the most important information. This technique is especially useful for creating compact word representations from large, sparse co-occurrence matrices.

#### SVD Decomposition of Matrix $X$
Given a matrix $X$ with dimensions $m \times n$, where:
   - $m$: number of words in the vocabulary
   - $n$: contexts or co-occurrence contexts

The **SVD decomposition** of $X$ can be expressed as:
$$
X = U \Sigma V^T
$$
where:
- $U$ is an $m \times k$ matrix of left singular vectors representing the words.
- $\Sigma$ is a $k \times k$ diagonal matrix with singular values $\sigma_1, \sigma_2, \ldots, \sigma_k$ on the diagonal.
- $V$ is an $n \times k$ matrix of right singular vectors representing the contexts.

This decomposition can also be represented as the sum of rank-1 matrices:
$$
X = \sigma_1 u_1 v_1^T + \sigma_2 u_2 v_2^T + \ldots + \sigma_k u_k v_k^T
$$
where:
- $\sigma_i$ are singular values indicating the importance of each component.
- $u_i v_i^T$ is a rank-1 matrix (an outer product of vectors $u_i$ and $v_i$) that captures a certain pattern in $X$.

Each $\sigma_i u_i v_i^T$ term contributes less information as $i$ increases, with the first few terms containing the most crucial information.

#### Truncated SVD for Dimensionality Reduction
To reduce dimensionality while retaining the most important information:
1. **Rank-1 Approximation**: Truncating at the first term $\sigma_1 u_1 v_1^T$ gives the best rank-1 approximation of $X$. This keeps the largest singular value, which represents the principal component (most significant feature) in the data.
2. **Rank-2 Approximation**: Including the first two terms $\sigma_1 u_1 v_1^T + \sigma_2 u_2 v_2^T$ provides the best rank-2 approximation, capturing more variation.
3. **Higher-Rank Approximations**: By retaining the top $k$ terms, we obtain a rank-$k$ approximation of $X$, representing the data in a lower-dimensional space while preserving the most informative structure.

### Why Truncated SVD Works: Information Retention
Using only the top $k$ components allows us to compress the data, approximating $X$ with $m \times k + k + n \times k$ values instead of $m \times n$, which is a much smaller number if $k \ll m, n$. The SVD theorem guarantees that:
- The vectors $u_1, v_1$ and singular value $\sigma_1$ store the most important information (similar to the principal components in PCA).
- Subsequent terms contribute less and less important information.

#### Analogy: Color Compression
Consider an analogy where colors are represented with 8 bits (e.g., different shades of green like "very light green," "light green," etc.). If we compress these colors into 4 bits, we retain the most important information:
- While some subtle differences are lost, we retain enough information to still recognize that these are shades of green.

In the same way, SVD keeps the most important features in word representations. By reducing the dimensions, it brings out latent similarities between words—similar to how colors become more distinct when represented with fewer but essential bits.

### Summary: Benefits of SVD for Word Representations
1. **Dimensionality Reduction**: Reduces high-dimensional co-occurrence data to lower-dimensional word representations.
2. **Latent Semantic Discovery**: Identifies underlying patterns, grouping words with similar meanings closer together in the reduced space.
3. **Efficient Storage and Computation**: Reduces computational complexity and memory usage, making it easier to work with large vocabularies.

By using truncated SVD, we get dense, compact, and semantically rich word vectors that capture essential similarities between words.



### Example of SVD for Learning Word Representations

Suppose we have a small co-occurrence matrix $X$ representing the word relationships in a toy corpus with a vocabulary of 4 words: "human," "machine," "system," and "user."

#### Co-Occurrence Matrix $X$
$$
X = 
\begin{bmatrix}
0 & 2 & 0 & 1 \\
2 & 0 & 1 & 0 \\
0 & 1 & 0 & 2 \\
1 & 0 & 2 & 0
\end{bmatrix}
$$
Each cell $X_{i,j}$ represents how many times the word in row $i$ co-occurs with the word in column $j$.

### Step 1: Performing SVD on $X$

1. **Decompose** $X$ using SVD:
   $$
   X = U \Sigma V^T
   $$
   where:
   - $U$: matrix of **left singular vectors** (word vectors).
   - $\Sigma$: **diagonal matrix** of singular values.
   - $V$: matrix of **right singular vectors** (context vectors).

   After performing SVD (actual computation steps omitted for simplicity), we get:

   $$
   U = 
   \begin{bmatrix}
   -0.5 & 0.5 & 0.7 & 0.1 \\
   -0.5 & -0.5 & -0.1 & 0.7 \\
   0.5 & -0.5 & 0.7 & -0.1 \\
   0.5 & 0.5 & -0.1 & -0.7 
   \end{bmatrix}, 
   \quad
   \Sigma = 
   \begin{bmatrix}
   3 & 0 & 0 & 0 \\
   0 & 2.5 & 0 & 0 \\
   0 & 0 & 1 & 0 \\
   0 & 0 & 0 & 0.5
   \end{bmatrix}, 
   \quad
   V = 
   \begin{bmatrix}
   -0.5 & -0.5 & 0.5 & 0.5 \\
   0.5 & -0.5 & -0.5 & 0.5 \\
   0.7 & -0.1 & 0.7 & -0.1 \\
   0.1 & 0.7 & -0.1 & -0.7
   \end{bmatrix}
   $$

2. **Interpret Singular Values in $\Sigma$**:
   - The values on the diagonal of $\Sigma$ represent the importance of each component.
   - Here, $3$ and $2.5$ are the largest values, indicating that the first two components carry the most important information.

### Step 2: Truncate for Dimensionality Reduction

To reduce dimensions, we can **truncate** $\Sigma$, $U$, and $V$ to keep only the top 2 singular values and corresponding vectors.

1. **Truncated Matrices**:
   $$
   U' = 
   \begin{bmatrix}
   -0.5 & 0.5 \\
   -0.5 & -0.5 \\
   0.5 & -0.5 \\
   0.5 & 0.5
   \end{bmatrix},
   \quad
   \Sigma' = 
   \begin{bmatrix}
   3 & 0 \\
   0 & 2.5
   \end{bmatrix},
   \quad
   V' = 
   \begin{bmatrix}
   -0.5 & -0.5 \\
   0.5 & -0.5 \\
   0.7 & -0.1 \\
   0.1 & 0.7
   \end{bmatrix}
   $$

2. **Reduced Representation of $X$**:
   Now, we approximate $X$ by multiplying these truncated matrices:
   $$
   X \approx U' \Sigma' V'^T
   $$

   This lower-rank approximation captures the core structure of $X$ using fewer dimensions.

### Step 3: Resulting Word Representations

Each row in the truncated $U' \Sigma'$ matrix gives us a **2-dimensional representation of each word** in the vocabulary:
   - **Word vectors**:
     - "human": $[-1.5, 1.25]$
     - "machine": $[-1.5, -1.25]$
     - "system": $[1.5, -1.25]$
     - "user": $[1.5, 1.25]$

These vectors now capture **latent similarities**:
   - "human" and "machine" are similar, as they have similar vector orientations.
   - "system" and "user" are also similar in this space.

---


### Step-by-Step Walkthrough of Co-Occurrence Matrix, Similarity Computation, and SVD for Word Representations

To illustrate this process, let's use a co-occurrence matrix for words in a corpus. We'll go through matrix calculations, dimensionality reduction with SVD, and how the result captures word similarities in a simplified form.

---

#### Co-Occurrence Matrix $X$
Consider a simplified co-occurrence matrix $X$, where each cell $X_{ij}$ represents the frequency of two words appearing close to each other in a sample corpus. 

Let's start with a sample co-occurrence matrix where we have four words: "human," "machine," "system," and "user."

$$
X = 
\begin{bmatrix}
0 & 2.944 & 0 & 2.25 & \dots & 0 \\
2.944 & 0 & 0 & 2.25 & \dots & 0 \\
0 & 0 & 0 & 1.15 & \dots & 1.84 \\
2.25 & 2.25 & 1.15 & 0 & \dots & 0 \\
\dots & \dots & \dots & \dots & \dots & \dots \\
0 & 0 & 1.84 & 0 & \dots & 0
\end{bmatrix}
$$

Here:
- **Rows** and **columns** represent words.
- Each cell represents co-occurrence counts between pairs of words.

---

### Step 1: Matrix Multiplication to Generate $X X^T$
The matrix $X X^T$ is computed by multiplying $X$ by its transpose. Each element $(X X^T)_{ij}$ in this matrix is the **dot product** of the vectors corresponding to words $i$ and $j$. It roughly indicates **cosine similarity** between the word pairs.

#### Resulting $X X^T$
The matrix $X X^T$ becomes:

$$
X X^T = 
\begin{bmatrix}
32.5 & 23.9 & 7.78 & 20.25 & \dots & 7.01 \\
23.9 & 32.5 & 7.78 & 20.25 & \dots & 7.01 \\
7.78 & 7.78 & 0 & 17.65 & \dots & 21.84 \\
20.25 & 20.25 & 17.65 & 36.3 & \dots & 11.8 \\
\dots & \dots & \dots & \dots & \dots & \dots \\
7.01 & 7.01 & 21.84 & 11.8 & \dots & 28.3
\end{bmatrix}
$$

Each cell $(X X^T)_{ij}$ captures similarity information between words $i$ and $j$, where higher values indicate greater similarity.

#### Cosine Similarity
For instance, the similarity between "human" and "user" can be computed as:

$$
\text{cosine\_sim}(\text{human, user}) = \frac{(X X^T)_{\text{human, user}}}{\|X_{\text{human}}\| \|X_{\text{user}}\|} = 0.21
$$

---

### Step 2: SVD on $X$ for Dimensionality Reduction

Now we apply Singular Value Decomposition (SVD) to $X$ to reduce its dimensions and retain only the most important information.

1. **SVD Decomposition**:
   - We decompose $X$ into three matrices:
     $$
     X = U \Sigma V^T
     $$
     where:
     - $U$ contains **left singular vectors** (word representations),
     - $\Sigma$ is a **diagonal matrix** of singular values,
     - $V$ contains **right singular vectors** (context representations).

2. **Truncate $\Sigma$**:
   To reduce dimensions, we can truncate $\Sigma$ to keep only the top singular values.

#### Low-Rank Approximation of $X$ After SVD
Using the truncated matrices, we reconstruct an approximate version of $X$:
$$
X_{\text{low rank}} = U' \Sigma' (V')^T
$$

---

### Step 3: Interpreting Low-Rank Representations

The low-rank matrix $X_{\text{low rank}}$ captures **latent relationships** between words:

$$
X_{\text{low rank}} = 
\begin{bmatrix}
2.01 & 2.01 & 0.23 & 2.14 & \dots & 0.43 \\
2.01 & 2.01 & 0.23 & 2.14 & \dots & 0.43 \\
0.23 & 0.23 & 1.17 & 0.96 & \dots & 1.29 \\
2.14 & 2.14 & 0.96 & 1.87 & \dots & -0.13 \\
\dots & \dots & \dots & \dots & \dots & \dots \\
0.43 & 0.43 & 1.29 & -0.13 & \dots & 1.71
\end{bmatrix}
$$

The lower dimensions allow us to see **latent semantic similarities**:
- For example, "human" and "machine" now have similar values, indicating that they are semantically related within this context.

---

This shows how **SVD extracts key relationships** from high-dimensional co-occurrence matrices and provides low-dimensional word representations that capture word similarities effectively. Each step from computing $X X^T$ to SVD decomposition and low-rank approximation reveals how related words are grouped together, uncovering **latent structures** in word usage patterns. 

---

### Problems with Co-Occurrence Matrices  

Although co-occurrence matrices capture word meanings to an extent, they also face several significant challenges:  

1. **High Dimensionality**:   
   - The matrix dimension is equal to the vocabulary size ($|V|$), which can be extremely large, especially with a substantial corpus.  
   - **Example**: For a vocabulary of 50,000 words, the matrix would have 50,000 rows and columns, resulting in a $50,000 \times 50,000$ matrix.  

2. **Sparsity**:  
   - Co-occurrence matrices are sparse because each word only co-occurs with a small subset of other words.  
   - **Result**: Most of the matrix cells contain zeroes, leading to memory inefficiency and computational challenges.  

3. **Growth with Vocabulary Size**:  
   - As the vocabulary size increases, the matrix dimension grows quadratically, making it increasingly difficult to store and process.  

### Example Co-Occurrence Matrix $X$ with Headers  

$$
X =   
\begin{array}{c|ccccc}  
       & \text{human} & \text{machine} & \text{system} & \text{for} & \text{user} \\
\hline  
\text{human}   & 2.01 & 2.01 & 0.23 & 2.14 & 0.43 \\
\text{machine} & 2.01 & 2.01 & 0.23 & 2.14 & 0.43 \\
\text{system}  & 0.23 & 0.23 & 1.17 & 0.96 & 1.29 \\
\text{for}     & 2.14 & 2.14 & 0.96 & 1.87 & -0.13 \\
\text{user}    & 0.43 & 0.43 & 1.29 & -0.13 & 1.71 \\
\end{array}  
$$

Each row and column is labeled, making it clear that:  
- Each entry represents the co-occurrence frequency or relationship between these specific words.  
- The diagonal values represent a word's co-occurrence with itself, often higher than off-diagonal values.  

### Solution: Dimensionality Reduction Using Singular Value Decomposition (SVD)  

To address these issues, **dimensionality reduction** techniques like Singular Value Decomposition (SVD) are commonly applied to co-occurrence matrices.  

1. **What is SVD?**  
   - SVD decomposes a matrix $X$ into three matrices:  
   $$
   X = U \Sigma V^T  
   $$
     - $U$: Matrix of the left singular vectors (word representations).  
     - $\Sigma$: Diagonal matrix with singular values, which indicate the importance of each component.  
     - $V$: Matrix of the right singular vectors (context representations).  
   - By keeping only the top $k$ singular values in $\Sigma$, we reduce the dimensionality of $X$.  

2. **Benefits of SVD for Word Representations**:  
   - **Lower Dimensionality**: Reduces the matrix from $|V|$-dimensional to $k$-dimensional, where $k$ is much smaller (e.g., 100–300).  
   - **Dense Representations**: Transforms sparse vectors into dense vectors, where each component has some meaningful information.  
   - **Capturing Semantics**: Helps capture latent semantic relationships, as similar words are now closer in the reduced vector space.  

3. **Example of Reduced Co-Occurrence Matrix**:  
   - After applying SVD, each word now has a lower-dimensional representation, say $\mathbb{R}^{300}$ instead of $\mathbb{R}^{50,000}$, making it computationally efficient and capturing more meaningful relationships.  

---

Here's the complete step-by-step walkthrough of the co-occurrence matrix, similarity computation, and SVD for word representations, including headers for clarity.

### Step-by-Step Walkthrough of Co-Occurrence Matrix, Similarity Computation, and SVD for Word Representations

To illustrate this process, we'll use a co-occurrence matrix for words in a corpus. We'll go through matrix calculations, dimensionality reduction with SVD, and how the result captures word similarities in a simplified form.

---

#### Co-Occurrence Matrix $X$

Consider a simplified co-occurrence matrix $X$, where each cell $X_{ij}$ represents the frequency of two words appearing close to each other in a sample corpus. 

Let's start with a sample co-occurrence matrix where we have five words: "human," "machine," "system," "for," and "user."

$$
X = 
\begin{array}{c|ccccc}
       & \text{human} & \text{machine} & \text{system} & \text{for} & \text{user} \\
\hline
\text{human} & 2.01 & 2.01 & 0.23 & 2.14 & 0.43 \\
\text{machine} & 2.01 & 2.01 & 0.23 & 2.14 & 0.43 \\
\text{system} & 0.23 & 0.23 & 1.17 & 0.96 & 1.29 \\
\text{for} & 2.14 & 2.14 & 0.96 & 1.87 & -0.13 \\
\text{user} & 0.43 & 0.43 & 1.29 & -0.13 & 1.71 \\
\end{array}
$$

Here:
- **Rows** and **columns** represent words.
- Each cell represents co-occurrence counts between pairs of words.

---

### Step 1: Matrix Multiplication to Generate $X X^T$

The matrix $X X^T$ is computed by multiplying $X$ by its transpose. Each element $(X X^T)_{ij}$ in this matrix is the **dot product** of the vectors corresponding to words $i$ and $j$. It roughly indicates **cosine similarity** between the word pairs.

#### Resulting $X X^T$

The matrix $X X^T$ becomes:

$$
X X^T = 
\begin{array}{c|ccccc}
       & \text{human} & \text{machine} & \text{system} & \text{for} & \text{user} \\
\hline
\text{human} & 25.4 & 25.4 & 7.6 & 21.9 & 6.84 \\
\text{machine} & 25.4 & 25.4 & 7.6 & 21.9 & 6.84 \\
\text{system} & 7.6 & 7.6 & 24.8 & 18.03 & 20.6 \\
\text{for} & 21.9 & 21.9 & 0.96 & 24.6 & 15.32 \\
\text{user} & 6.84 & 6.84 & 20.6 & 15.32 & 17.11 \\
\end{array}
$$

Each cell $(X X^T)_{ij}$ captures similarity information between words $i$ and $j$, where higher values indicate greater similarity.

#### Cosine Similarity

For instance, the similarity between "human" and "user" can be computed as:

$$
\text{cosine\_sim}(\text{human, user}) = \frac{(X X^T)_{\text{human, user}}}{\|X_{\text{human}}\| \|X_{\text{user}}\|} = 0.21
$$

---

### Step 2: SVD on $X$ for Dimensionality Reduction

Now we apply Singular Value Decomposition (SVD) to $X$ to reduce its dimensions and retain only the most important information.

1. **SVD Decomposition**:
   - We decompose $X$ into three matrices:
     $$
     X = U \Sigma V^T
     $$
     where:
     - $U$ contains **left singular vectors** (word representations),
     - $\Sigma$ is a **diagonal matrix** of singular values,
     - $V$ contains **right singular vectors** (context representations).

2. **Truncate $\Sigma$**:
   To reduce dimensions, we can truncate $\Sigma$ to keep only the top singular values.

---

### Step 3: Interpreting Low-Rank Representations

The low-rank matrix $X_{\text{low rank}}$ captures **latent relationships** between words:

$$
X_{\text{low rank}} = 
\begin{array}{c|ccccc}
       & \text{human} & \text{machine} & \text{system} & \text{for} & \text{user} \\
\hline
\text{human} & 2.01 & 2.01 & 0.23 & 2.14 & 0.43 \\
\text{machine} & 2.01 & 2.01 & 0.23 & 2.14 & 0.43 \\
\text{system} & 0.23 & 0.23 & 1.17 & 0.96 & 1.29 \\
\text{for} & 2.14 & 2.14 & 0.96 & 1.87 & -0.13 \\
\text{user} & 0.43 & 0.43 & 1.29 & -0.13 & 1.71 \\
\end{array}
$$

The lower dimensions allow us to see **latent semantic similarities**:
- For example, "human" and "machine" now have similar values, indicating that they are semantically related within this context.

---

### Final Representation of Words and Contexts

The resulting matrices provide a more efficient representation:
- $W_{\text{word}} = U$ (word representation): captures the reduced-dimensional representation of each word in the vocabulary.
- $W_{\text{context}} = V$ (context representation): captures the reduced-dimensional representation of the context of each word.

By taking the **row vectors** from $W_{\text{word}}$ for each word, we obtain the **low-dimensional word embeddings** that capture semantic similarity while maintaining the relative positions and similarities as in the original high-dimensional space.

---
### Understanding the Relationships in SVD

When we apply Singular Value Decomposition (SVD) to the co-occurrence matrix $X$, we express it as follows:

$X = U \Sigma V^T$

Where:
- $U$ is the matrix of left singular vectors, which corresponds to the **word representations**. Each row in $U$ represents a word in a lower-dimensional space.
- $\Sigma$ is a diagonal matrix containing the singular values that indicate the importance of each corresponding dimension.
- $V^T$ is the transpose of the matrix of right singular vectors, which corresponds to **context representations**.

### Step-by-Step Breakdown of the Dot Product Relationship

#### 1. **Matrix Multiplication of $X$**
When we multiply $X$ by its transpose $X^T$, we get:

$X X^T = (U \Sigma V^T)(U \Sigma V^T)^T$

Using the property of transposes, we can rewrite the right-hand side:

$= (U \Sigma V^T)(V \Sigma^T U^T)$

Since $\Sigma$ is diagonal, its transpose $\Sigma^T$ is just $\Sigma$, leading to:

$= U \Sigma V^T V \Sigma^T U^T$

Here, $V^T V = I$ (the identity matrix), simplifying the equation further:

$= U \Sigma \Sigma^T U^T$

#### 2. **Recognizing the Dot Product of $W_{\text{word}}$**
The product $\Sigma \Sigma^T$ is a matrix that contains the squares of the singular values along its diagonal. Thus, we have:

$X X^T = U (\Sigma \Sigma^T) U^T$

This shows that the dot product between the rows of $U$ (i.e., $W_{\text{word}}$) is equivalent to the dot product between the words represented in the original co-occurrence matrix $X$.

#### 3. **Implications for Word Representations**
Since $W_{\text{word}} = U$, the relationship can be articulated as:

$W_{\text{word}} W_{\text{word}}^T = U (U^T)$

This means:
- Each element in $W_{\text{word}} W_{\text{word}}^T$ reflects the similarity (via the dot product) between the word embeddings of the vocabulary words.
- Thus, the structure of the co-occurrence matrix $X X^T$ is preserved in the lower-dimensional representation $W_{\text{word}} W_{\text{word}}^T$.

#### 4. **Final Representation**
Concisely, we can summarize the results:
- **Word Representations**: The matrix $W_{\text{word}} = U$ captures the embeddings of words in a reduced-dimensional space.
- **Context Representations**: The matrix $W_{\text{context}} = V$ captures the embeddings of context words.

### Conclusion
The relationship established here shows that by using SVD, we can derive a low-dimensional representation of word embeddings that retains the original co-occurrence information from the matrix $X$. The dot products of the rows of $W_{\text{word}}$ directly correspond to the similarities captured in the original co-occurrence matrix $X$, allowing us to leverage this structure for various natural language processing tasks. 

---


### Continuous Bag of Words (CBOW) Model

#### 1. **Overview of Word Representation Models**
   - **Count-Based Models**: 
     - Previous methods discussed (like co-occurrence matrices) are known as count-based models.
     - They utilize the frequency of word co-occurrences to derive representations, focusing on how often words appear together in a given context.
   - **Prediction-Based Models**: 
     - The CBOW model is part of a family of methods that learn word representations by predicting a target word based on its context.
     - These models focus on understanding the relationships between words through prediction tasks rather than mere counting.

#### 2. **Objective of CBOW**
   - The primary objective of the CBOW model is to predict a target word based on the surrounding context words (previous words).
   - This is done by using the context words to generate a representation for the target word, effectively learning its meaning in the process.

#### 3. **Training Data Construction**
   - For training the CBOW model, we utilize "n-word windows" from the corpus. 
   - **Example Task**: Predict the n-th word given the previous n-1 words. 
     - For instance, given the sentence: "he sat on a chair," if we take a window of size 2 (n=2):
       - **Input Context**: "he" (1st word)
       - **Target Word**: "sat" (2nd word)
   - The training data can be constructed by sliding a window over the entire corpus (e.g., Wikipedia), capturing all possible n-word combinations.

#### 4. **CBOW Model Mechanics**
   - The CBOW architecture consists of the following components:
     - **Input Layer**: Represents context words (previous n-1 words).
     - **Hidden Layer**: The model does not have an explicit activation function; it averages the vector representations of the context words.
     - **Output Layer**: The output is a probability distribution over the vocabulary, predicting the target word.
   - **Training Process**:
     - During training, the model adjusts the weights based on the error between predicted and actual target words.
     - This process uses techniques like backpropagation and stochastic gradient descent to minimize prediction errors.

#### 5. **Example of Context and Target**
   - From the sentence "he sat on a chair," and taking a context of size 1 (n=2):
     - **Context**: "he" (input)
     - **Target**: "sat" (output)
   - Another example could be:
     - **Context**: "sat" (input)
     - **Target**: "on" (output)

#### 6. **Benefits of CBOW**
   - **Efficient Learning**: CBOW can learn from a large amount of data, capturing semantic meanings effectively.
   - **Low-Dimensional Representations**: By predicting target words, the model generates lower-dimensional embeddings that encapsulate meaning, allowing for easier comparisons between words.
   - **Handling Large Vocabulary**: The softmax output layer allows for manageable probabilities over large vocabularies, making it suitable for tasks with extensive datasets.

#### 7. **Conclusion**
   - The CBOW model represents a significant advancement in natural language processing, transitioning from simple counting methods to more sophisticated predictive modeling.
   - By learning to predict target words based on context, CBOW helps capture semantic relationships between words, paving the way for more complex applications in language understanding.


### Modeling with a Feedforward Neural Network

#### 1. **Problem Statement**
   - The goal is to predict a target word based on a given context word using a feedforward neural network. 
   - **Input**: One-hot representation of the context word.
   - **Output**: A probability distribution over all words in the vocabulary, predicting the likelihood of each word being the target.

#### 2. **Input Representation**
   - Each context word is represented as a **one-hot vector**.
     - For a vocabulary of size $V$, the one-hot vector for a word will have $V$ dimensions with all values set to 0 except for a single position (1) corresponding to the word's index in the vocabulary.
     - For example, if "sat" is the 3rd word in the vocabulary, its one-hot representation will be:
       $\text{one-hot("sat")} = [0, 0, 1, 0, \ldots, 0]$ $(V \text{ dimensions})$

#### 3. **Neural Network Architecture**
   - **Input Layer**: 
     - The one-hot vector is fed into the network.
   - **Hidden Layer**:
     - The one-hot vector is multiplied by a weight matrix $W_{\text{context}}$ (dimensions $R^{k \times V}$), where $k$ is the dimension of the hidden layer.
     - This produces a hidden layer representation:
       $h = W_{\text{context}} \cdot \text{one-hot(context\_word)}$ $(h \in R^k)$
   - **Output Layer**:
     - The hidden representation $h$ is then passed through another weight matrix $W_{\text{word}}$ (dimensions $R^{V \times k}$), producing a score vector for each word in the vocabulary.
       $\text{score} = W_{\text{word}} \cdot h$ $(\text{score} \in R^V)$
     - This score vector represents the unnormalized log probabilities for each word.

#### 4. **Probability Distribution**
   - To convert the score vector into a probability distribution over the vocabulary, we apply the **softmax** function:
     $P(w) = \frac{e^{\text{score}_w}}{\sum_{j=1}^{V} e^{\text{score}_j}}$
   - Here, $P(w)$ is the probability of predicting word $w$ given the context.

#### 5. **Training Process**
   - The model is trained using labeled data, where the context words and their corresponding target words are known.
   - A loss function (e.g., cross-entropy loss) is computed based on the predicted probabilities and the true target word. The model parameters $W_{\text{context}}$ and $W_{\text{word}}$ are updated using optimization algorithms (like SGD) to minimize this loss.

#### 6. **Parameter Matrix Dimensions**
   - $W_{\text{context}}$: This weight matrix has dimensions $R^{k \times V}$, mapping the input one-hot vector to the hidden layer representation.
   - $W_{\text{word}}$: This weight matrix has dimensions $R^{V \times k}$, mapping the hidden layer representation to the output probabilities for each word.

#### 7. **Example Calculation**
   - Suppose we want to predict the word "chair" given the context word "sat":
     - **Input (one-hot)**: 
       $\text{one-hot("sat")} = [0, 0, 1, 0, \ldots, 0]$
     - Compute hidden representation:
       $h = W_{\text{context}} \cdot \text{one-hot("sat")}$
     - Compute scores:
       $\text{score} = W_{\text{word}} \cdot h$
     - Compute probabilities for all words:
       $P(\text{chair}) = \frac{e^{\text{score}_{\text{chair}}}}{\sum_{j=1}^{V} e^{\text{score}_j}}$



### Understanding the Probability Calculation $P(\text{on | sat})$

#### 1. **Probability of the Target Word**
   - The probability $P(\text{on | sat})$ is calculated using the dot product between the word embedding matrices:
     $P(\text{on | sat}) = \frac{e^{(W_{\text{word}}h)[i]}}{\sum_{j} e^{(W_{\text{word}}h)[j]}}$
   - Here:
     - $W_{\text{word}}$ is the weight matrix corresponding to the target words.
     - $h$ is the hidden state derived from the context word's one-hot representation.
     - $(W_{\text{word}}h)[i]$ indicates the dot product involving the $i$-th column of $W_{\text{word}}$.

#### 2. **Softmax Function**
   - The softmax function is used to transform the scores (dot products) into a probability distribution over the vocabulary:
     - It normalizes the exponentiated scores so that they sum up to 1.
   - This allows us to interpret the output as a probability for each possible word in the vocabulary.

#### 3. **Representation of the Target Word**
   - The probability $P(\text{word} = \text{on | sat})$ depends on the $i$-th column of $W_{\text{word}}$, treating it as the embedding or representation of the target word.

### Relation to SVD (Singular Value Decomposition)
   - In the context of SVD, we noted that the columns of the word matrix corresponded to the representations of each word.
   - Similarly, in the CBOW model, each column of $W_{\text{word}}$ can be viewed as the word's representation, akin to how SVD represents data.

### Interpreting $W_{\text{context}}$ and One-Hot Vector $x$
   - The context matrix $W_{\text{context}}$ has dimensions $k \times |V|$ (where $k$ is the embedding dimension and $|V|$ is the vocabulary size). Each column corresponds to the embedding of a word.
   - When multiplying $W_{\text{context}}$ by the one-hot vector $x$, you extract the specific word representation based on the position of the '1' in $x$.

#### 4. **Matrix Multiplication $W_{\text{context}} \cdot x$**
   - Suppose $x$ is a one-hot vector, e.g., $[0, 1, 0]$ for the word at index 2:
     $W_{\text{context}} = \begin{bmatrix}
     -1 & 0.5 & 2 \\
     3 & -1 & -2 \\
     -2 & 1.7 & 3
     \end{bmatrix}$
   - The multiplication $W_{\text{context}} \cdot x$ will result in:
     $W_{\text{context}} \cdot \begin{bmatrix}
     0 \\
     1 \\
     0
     \end{bmatrix} = \begin{bmatrix}
     0.5 \\
     -1 \\
     1.7
     \end{bmatrix}$
   - This selects the 2nd column of $W_{\text{context}}$, which represents the embedding of the context word.

### Interpretation
- The output of $W_{\text{context}} \cdot x$ is the representation of the context word in the embedding space.
- This selected representation is then used to compute the hidden state $h$ and subsequently determine the probabilities of target words through the neural network.

### Summary of Key Points
- **One-Hot Vector**: Acts as a selector to choose a specific word representation from the context matrix.
- **Weight Matrices**: Each column of $W_{\text{word}}$ and $W_{\text{context}}$ represents the respective word's embedding.
- **Softmax**: Converts scores into a probability distribution for multi-class classification.
- **Analogy with SVD**: Both models share the concept that columns of the matrices correspond to word representations.

---

### Key Components of the CBOW Model

1. **Context Word and Target Word**
   - Let the context word be denoted by the index $c$ (e.g., "sat") and the target (output) word by the index $w$ (e.g., "on").
   - The goal is to predict the target word $w$ given the context word $c$.

2. **Hidden State Representation**
   - The hidden state $h$ is derived from the context word representation:
     $$
     h = W_{\text{context}} \cdot x_c = u_c
     $$
   - Here, $x_c$ is the one-hot vector representation of the context word $c$, and $u_c$ is the corresponding column of the $W_{\text{context}}$ matrix.

### Output Function

3. **Softmax Function**
   - For a multi-class classification problem, the appropriate output function $y = f(x)$ is the **softmax** function.
   - The softmax function computes the probability distribution over the vocabulary $V$ for the target word $w$:
     $$
     P(w | c) = y_w = \frac{\exp(u_c \cdot v_w)}{\sum_{j \in V} \exp(u_c \cdot v_j)}
     $$
   - In this equation:
     - $u_c$ is the column of $W_{\text{context}}$ corresponding to the context word $c$.
     - $v_w$ is the column of $W_{\text{word}}$ corresponding to the target word $w$.
     - The softmax function ensures that the output probabilities sum to 1.

### Loss Function

4. **Cross-Entropy Loss**
   - An appropriate loss function for this multi-class classification problem is **cross-entropy**.
   - The cross-entropy loss $L$ can be expressed as:
     $$
     L(\theta) = -\log P(w | c) = -\log y_w
     $$
   - Here, $y_w$ is the predicted probability of the target word $w$ given the context word $c$.

### Summary of Mathematical Representations

- **Hidden State Calculation**:
  $$
  h = W_{\text{context}} \cdot x_c = u_c
  $$

- **Softmax Probability**:
  $$
  P(w | c) = y_w = \frac{\exp(u_c \cdot v_w)}{\sum_{j \in V} \exp(u_c \cdot v_j)}
  $$

- **Cross-Entropy Loss**:
  $$
  L(\theta) = -\log P(w | c) = -\log y_w
  $$

### Explanation of Terms
- $W_{\text{context}}$: Weight matrix for context words; each column represents a context word's embedding.
- $W_{\text{word}}$: Weight matrix for target words; each column represents a target word's embedding.
- $u_c$: The embedding of the context word $c$ obtained from $W_{\text{context}}$.
- $v_w$: The embedding of the target word $w$ obtained from $W_{\text{word}}$.



### Accessing Columns in the Weight Matrix

1. **Matrix Representation**
   - Let $W$ be the weight matrix where:
     - $W \in \mathbb{R}^{k \times V}$
     - $k$ is the dimensionality of the word embeddings.
     - $V$ is the size of the vocabulary.
   - Each column $W[:, j]$ corresponds to the embedding of the $j$-th word in the vocabulary.

2. **Identifying Words**
   - If $i$ is the index of the word "he" in the vocabulary and $j$ is the index of the word "sat," you would access their respective columns in the weight matrix $W$:
     - $W[:, i]$: The column corresponding to the word "he."
     - $W[:, j]$: The column corresponding to the word "sat."

3. **Updating Context and Target Word Embeddings**
   - During training, when the context word "he" (represented by index $i$) and the target word "sat" (represented by index $j$) are used, the update rule for the target word embedding $v_j$ can be expressed as:
     $$
     v_j \leftarrow v_j + \eta \cdot u_c \cdot (1 - y_j)
     $$
   - Where:
     - $u_c = W[:, j]$ is the embedding for the context word.
     - $y_j$ is the predicted probability for the target word "sat."
     - $\eta$ is the learning rate.

### Efficient Computation

4. **Accessing Columns for Updates**
   - Instead of performing a full matrix multiplication, you can directly access the columns for the updates:
     - **For context:** Use $W[:, i]$ to access the embedding of "he."
     - **For target:** Use $W[:, j]$ to access the embedding of "sat."
   - The operation can be expressed in a more concise form:
     $$
     W[:, i] \text{ (context word)} + W[:, j] \text{ (target word)} \quad \text{(when updating)}
     $$

### Example
- Suppose you have the following embedding matrix $W$:
  $$
  W = \begin{bmatrix}
  0.1 & 0.2 & 0.3 \\
  0.4 & 0.5 & 0.6 \\
  0.7 & 0.8 & 0.9 \\
  \end{bmatrix}
  $$
- If "he" is at index 0 and "sat" is at index 1, the columns you access would be:
  - For "he": $W[:, 0] = \begin{bmatrix} 0.1 \\ 0.4 \\ 0.7 \end{bmatrix}$
  - For "sat": $W[:, 1] = \begin{bmatrix} 0.2 \\ 0.5 \\ 0.8 \end{bmatrix}$

5. **Performing Updates**
   - If you want to update the embedding of "sat" based on the context word "he," you can do so directly:
     $$
     v_j \leftarrow v_j + \eta \cdot W[:, i] \cdot (1 - y_j)
     $$

### Background Recap
1. **Model Architecture**
   - We are using a feedforward neural network where:
     - The input is a one-hot encoded vector representing the context words.
     - The output is the probability distribution over the vocabulary for the target word.

2. **Softmax Function**
   - The softmax function computes probabilities for each word in the vocabulary based on the scores from the linear transformation:
     $$
     P(w) = \frac{e^{u_c^T v_w}}{\sum_{j} e^{u_c^T v_j}}
     $$
   - Here, $u_c$ is the embedding for the context, $v_w$ is the embedding for the target word, and the denominator sums over all words in the vocabulary $V$.

3. **Loss Function**
   - The loss function for the multi-class classification problem using cross-entropy is:
     $$
     L = -\log(P(w))
     $$
   - We need to compute the gradient of this loss with respect to the parameters $W_{word}$ and $W_{context}$.

### Deriving the Update Rule for $v_w$
Let's focus on deriving the update rule for $v_w$ during backpropagation.

1. **Gradient of the Loss Function**
   - The gradient of the loss with respect to the predicted probability $P(w)$:
     $$
     \frac{\partial L}{\partial P(w)} = -\frac{1}{P(w)}
     $$
   - Now we need to use the chain rule to find the gradient with respect to $v_w$:
     $$
     \frac{\partial L}{\partial v_w} = \frac{\partial L}{\partial P(w)} \cdot \frac{\partial P(w)}{\partial v_w}
     $$

2. **Computing the Derivative of the Softmax Output**
   - The derivative of the softmax output can be expressed as:
     $$
     \frac{\partial P(w)}{\partial v_k} = P(w)(\delta_{w,k} - P(k))
     $$
   - Where $\delta_{w,k}$ is the Kronecker delta, which is 1 if $w = k$ and 0 otherwise.

3. **Combining the Derivatives**
   - Substituting back, we have:
     $$
     \frac{\partial L}{\partial v_k} = -\frac{1}{P(w)} \cdot P(w)(\delta_{w,k} - P(k)) = \delta_{w,k} - P(k)
     $$

4. **Gradient for the Update Rule**
   - The update for $v_w$ can be computed as:
     $$
     v_w \leftarrow v_w - \eta (\delta_{w,k} - P(k)) \cdot u_c
     $$
   - This means:
     - If $w$ is the correct word (i.e., $\delta_{w,k} = 1$), the update will be $v_w$ unchanged.
     - If $w$ is not the correct word (i.e., $\delta_{w,k} = 0$), $v_w$ will be updated to move closer to $u_c$.

### Summary of Key Points
- The gradient for the target word embedding $v_w$ is adjusted based on the difference between the predicted probability and the actual label.
- The update rule modifies $v_w$ depending on whether the predicted class matches the actual class, emphasizing the context embedding $u_c$.
- The softmax function's computational complexity is significant, as it requires summing over all vocabulary entries, leading to efficiency concerns in practice, particularly with large vocabularies.

### Practical Considerations
- In practice, to mitigate the computational burden of the softmax function during training, techniques like **negative sampling** or **hierarchical softmax** are often employed.
- These techniques reduce the number of computations required by only focusing on a subset of the vocabulary, improving training efficiency while still allowing for effective embedding learning.
