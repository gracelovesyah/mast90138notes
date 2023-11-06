# Week 3 Lecture 1

## 5 Principal Component Analysis (PCA)

### 5.1 Introduction
- **Combining Variables**: For instance, given age and height, PCA enables us to merge these variables into a singular, new composite feature.

### 5.2 Detailed Overview of PCA
- **Objective**: The purpose of PCA is to identify the line that best captures the spread of the data, or formally, the linear projection that maximizes variance.
  
- **Projection Formula**: The new component $ Y_{i1} $ for a data point $ X_i $ is a weighted sum of its original variables, $ Y_{i1} = a^T X_i $, where $ a $ is the direction vector of the projection and is a unit vector ($ \|a\|^2 = 1 $).

- **Eigenvalues and Eigenvectors**:
  - Consider $ \gamma_1, \ldots, \gamma_p $ as the normalized eigenvectors (each with norm 1) of the covariance matrix $ \Sigma $, corresponding to the eigenvalues $ \lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_p $.
  - Note that the direction of $ \gamma_j $ is not uniquely determined; it can be flipped ($ \gamma_j $ can be replaced with $ -\gamma_j $) without affecting the analysis.
  - The optimal $ a $ that maximizes the variance of the projected data $ Y_{i1} $ is equal to $ \gamma_1 $, which is associated with the largest eigenvalue.

- **First Principal Component**:
  - The first principal component $ Y_{i1} $ is the projection onto $ \gamma_1 $, which can be computed as $ Y_{i1} = \gamma_1^T X_i $ for the original variables.
  - If the dataset is not centered, $ Y_{i1} $ is adjusted to be $ Y_{i1} = \gamma_1^T (X_i - \mu) $, where $ \mu $ is the mean of $ X_i $.
  - This first principal component represents the direction along which the data varies the most.
  - Standard practice involves centering the data by subtracting the mean before computing the principal components.

- **Subsequent Principal Components**:
  - Subsequent principal components are found by identifying new axes that are orthogonal to the previously determined ones and that capture the maximum remaining variance.
  - Each principal component $ Y_{ij} $ has an expected value of 0 and a variance $ \lambda_j $, with each $ Y_{ij} $ being uncorrelated to the others ($ \text{cov}(Y_{ik}, Y_{ij}) = 0 $ for $ k \neq j $).
  - The variances of the principal components are ordered such that $ \text{var}(Y_{i1}) \geq \text{var}(Y_{i2}) \geq \ldots \geq \text{var}(Y_{ip}) $.
  - The sum of the variances of all the principal components equals the trace of the covariance matrix $ \Sigma $, and the product of these variances is equal to the determinant of $ \Sigma $.

By redefining the data in terms of these principal components, PCA simplifies the complexity of multidimensional data while retaining the aspects that contribute most to its variance.