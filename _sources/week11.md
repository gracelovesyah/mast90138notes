# week11. lec1
This note is completed with the assistance of [ChatGPT](https://chat.openai.com/c/7579ad2e-6406-4ef1-b07d-a24466c1f587)

### Out of Bag 
- Additional Resource
    - [Wikipedia](https://en.wikipedia.org/wiki/Out-of-bag_error)
    - [towardsdatascience](https://towardsdatascience.com/what-is-out-of-bag-oob-score-in-random-forest-a7fa23d710)



**Multivariate Statistics for Data Science - Detailed Lecture Summary:**

### **1. Bagging in Decision Trees:**
- **Concept:** Bagging (Bootstrap Aggregating) involves creating multiple bootstrap trees and aggregating their results to enhance accuracy and reduce variance.
- **Key Points:**
  - The aggregated classifier isn't a single tree but can outperform individual trees.
  - Bagging reduces variance without increasing bias.
  - Example: By bagging with varying values of B (number of bootstrap samples), one can observe its effect on performance, such as classification error.

### **2. Random Forests for Classification:**
- **Concept:** An enhancement over bagging, random forests aim to decorrelate individual trees to further reduce variance.
- **Key Points:**
  - Trees in bagging aren't independent; random forests address this by selecting a random subset of predictors for each split.
  - The default number of predictors considered at each split is $ m = \sqrt{p} $, but this can be tuned.
  - The algorithm involves:
    - Drawing bootstrap samples.
    - Growing trees by selecting random predictors and making splits.
    - Aggregating results for predictions, either by averaging (regression) or majority voting (classification).

### **3. Variable Importance in Decision Trees & Random Forests:**
- **Concept:** Not all predictors contribute equally to the model's decision-making. Their importance can be quantified.
- **Key Points:**
  - Trees split data based on predictors, ideally focusing on the most informative ones.
  - Importance can be measured by:
    - The decrease in node impurity (often using the Gini impurity).
    - The decrease in prediction accuracy when a predictor's values are permuted (OOB method).
  - In Random Forests, the importance of a predictor is averaged across all trees.
  - Importance scores are often normalized, with higher values indicating more important predictors.

### **4. Implementing Random Forests in R:**
- **Concept:** Practical steps to compute a random forest in R.
- **Key Points:**
  - The `randomForest` package in R is used.
  - Data input involves specifying a formula, indicating the response variable and predictors.
  - The response variable should be formatted as a factor for classification tasks.
  - By default, the package uses $ m = \sqrt{p} $ predictors for each split in classification tasks.

---

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
