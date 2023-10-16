# week11. lec2
**Slide Summary: MAST 90138: MULTIVARIATE STATISTICAL TECHNIQUES**

---

**1. Cluster Analysis Introduction:**
- **Classification vs. Clustering**:
  - Classification: Classify into known groups with labeled training data.
  - Clustering: Identify potential groups without labeled data (unsupervised learning).
- Data Format: Observations $X_1, ..., X_n$, with $X_i = (Xi_1, ..., Xi_p)^T$.

---

**2. Real-life Example**:
- A new company wishes to identify clusters within their customers based on purchasing behavior. No training data is available.

---

**3. Clustering Objectives**:
- Individuals within clusters should be more closely related to each other than to those in other clusters.
- Hierarchical clustering: Arrange clusters in a hierarchy, breaking down larger clusters into smaller ones.

---

**4. Principles of Cluster Analysis**:
- Used to determine if observations come from multiple groups.
- Individuals within a cluster are similar. This similarity depends on the defined measure.
- Selecting the right similarity measure is crucial and should align with the data type and problem at hand.

---

**5. Dissimilarity Matrices**:
- Used by many clustering algorithms. Matrix $D$ such that $D_{ij}$ measures the dissimilarity between the $i^{th}$ and $j^{th}$ individuals.
- Requirements:
  - Nonnegative elements.
  - Zero diagonal elements: $D_{ii} = 0$.
  - Symmetric, if not, replace with $(D + D^T)/2$.

---

**6. Understanding 'Distance' in Clustering**:
- Dissimilarity can be seen as a function $D : \mathbb{R}^p \times \mathbb{R}^p \rightarrow \mathbb{R}^+$.
- Real distances satisfy specific properties like symmetry $D(a, b) = D(b, a)$ and the triangle inequality.

---

**7. Measuring Similarity using "Correlation"**:
- A method measures the similarity between two individuals, $i$ and $k$, with the formula: 

$$ \rho(X_i, X_k) = \frac{\sum_{j=1}^{p}(X_{ij} - \bar{X}_i)(X_{kj} - \bar{X}_k)}{\sqrt{\sum_{j=1}^{p}(X_{ij} - \bar{X}_i)^2 \sum_{j=1}^{p}(X_{kj} - \bar{X}_k)^2}} $$

- Transform this similarity into dissimilarity using: $ D_{ik} = 1 - \rho(X_i, X_k) $.

---

**8. Types of Data and Their Treatment**:
- **Categorical/Nominal Variables**: No inherent order. Users need to define a custom measure for differences.
  
- **Ordinal Variables**: Have an inherent order or rank. They can be transformed to mimic quantitative variables for clustering. For $M$ distinct ordered values:

$$ \text{Transformed Value} = \frac{i - 1/2}{M} $$

---