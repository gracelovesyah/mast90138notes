# week8 lec 2

p.228 - p.244

This note is completed with the assistance of [ChatGPT](https://chat.openai.com/c/c1fcc91e-4844-44f7-93e6-1f33b85d11f8)


---

**Week 8, Lecture 2: Binary Classification using Regression**

1. **Binary Classification**:
    - Observed training samples, e.g., determining two types of leukemia (ALL and AML) based on gene expressions (X1, X2).
    - Data can be represented using indicator functions: $ Y_{ik} $ (derived from $ G_i $).
    - Probability of belonging to group k given X is $ P(G=k|X=x) = m_k(x) $.
    - $ m_2(x) = 1 - m_1(x) $. Hence, only need to estimate $ m_1 $.

2. **Linear Regression Classifier**:
    - Assumes $ m_1(x) = E(Y_1|X=x) = \beta_0 + \beta^T x $.
    - Estimates $ m_1 $ using least squares, PCA, or PLS for dimension reduction.
    - Classification boundary is $ \beta_0 + \beta^T x = 1/2 $.
    - Limitation: Estimated probability might not lie between 0 and 1.

3. **Logistic Regression Classifier**:
    - Assumes $ m_1(x) = E(Y_1|X=x) = \frac{exp(\beta_0 + \beta^T x)}{1 + exp(\beta_0 + \beta^T x)} $.
    - Uses maximum likelihood to estimate parameters.
    - Classification boundary is $ \beta_0 + \beta^T x = 0 $.
    - Probabilities always between 0 and 1, making it ideal for binary classification.

4. **Comparison**:
    - Logistic regression is often preferred for binary classification due to its range of outputs and assumptions more aligned with binary outcomes. Linear regression can be used but has limitations, especially when it comes to representing probabilities.

5. **Practical Examples**:
    - Golub et al. (1999): Classification of two types of leukemia based on gene expressions.
    - For logistic model estimation: $ x_2 = 0.08803435 + 0.05121337x_1 $.

**Key Takeaways**:
- Logistic regression is more appropriate for binary classification compared to linear regression, especially when estimating probabilities.
- Linear regression has its place but might not always be the best tool for binary classification.

---