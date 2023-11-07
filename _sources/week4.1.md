# week4 lec 1

## PCA cont.
### 5.3 In practice

In practice we do not know Σ nor µ = E(Xi) and instead we use their empirical counterparts ¯X and S, i.e.:


Yi1 = gT
(Xi − ¯X )
1


Yik = gT
(Xi − ¯X )
k

note, x1 and x bar are vectors
In what follows I will not always use empirical notations as that can quickly become too heavy but IN PRACTICE WE ALWAYS USE EMPIRICAL VERSIONS. 

```{admonition} Example: Swiss bank notes data 真假钞票例子💵
X1: Length of bill (mm)
X2: Width of left edge (mm)
X3: Width of right edge (mm)
X4: Bottom margin width (mm)
X5: Top margin width (mm)
X6: Length of diagonal (mm)
The first 100 banknotes are genuine and the next 100 are counterfeit.

$$
 Y = \Gamma^T(X_i - \mu)
$$


1. $ \text{E}(X_j) = 0 $ for $ j = 1, \ldots, p $:
   - This means that the expected value (mean) of the variables $ X_j $ is zero. In PCA, it's a common practice to center the data by subtracting the mean of each variable from the dataset to ensure that the first principal component describes the direction of maximum variance.

2. $ \text{var} (Y_{ij}) = \lambda_j $ for $ j = 1, \ldots, p $:
   - Here, $ \text{var} (Y_{ij}) $ is the variance of the $ j $-th principal component, and $ \lambda_j $ represents its corresponding eigenvalue. The variance of each principal component is equal to its eigenvalue, and PCA aims to maximize the variance explained by each principal component.

3. $ \text{cov} (Y_{ik}, Y_{ij}) = 0 $, for $ k \neq j $:
   - The covariance between any two different principal components is zero. This implies that the principal components are orthogonal to each other, meaning they are uncorrelated and each represents a different source of variance in the data.

4. $ \text{var} (Y_{i1}) \geq \text{var} (Y_{i2}) \geq \ldots > \text{var} (Y_{in}) $:
   - The variances of the principal components are ordered in a non-increasing fashion. This property ensures that the first few principal components capture most of the variability in the data, which is why PCA is useful for reducing dimensionality.



5. $ \sum Y_{ij} = \text{tr}(\Sigma) $:
   - This means that the sum of the eigenvalues ($Y_{ij}$) of the covariance matrix ($\Sigma$) is equal to the trace of the covariance matrix. The trace of a matrix is the sum of its diagonal elements, which, in the case of a covariance matrix, are the variances of each variable. In PCA, the total variance captured by all the principal components is equal to the sum of the variances of the original variables.

6. $ \prod Y_{ij} = |\Sigma| $:
   - This states that the product of the eigenvalues ($Y_{ij}$) of the covariance matrix ($\Sigma$) is equal to the determinant of the covariance matrix. The determinant of the covariance matrix can be interpreted as a measure of the overall variance in the multivariate data. It is also related to the volume of the confidence ellipsoid in the multivariate normal distribution.

```

---

```{admonition} Proof: $ \text{E}(X_j) = 0 $ for $ j = 1, \ldots, p $
:class: dropdown
The expectation operator $ \text{E} $ is a linear operator, which means it satisfies the following properties:

1. **Additivity**: $ \text{E}(X + Y) = \text{E}(X) + \text{E}(Y) $
2. **Homogeneity**: $ \text{E}(cX) = c\text{E}(X) $

Where $ X $ and $ Y $ are random variables, and $ c $ is a constant.

Given a matrix $ \Gamma $ and a random vector $ X $, the transformed random vector $ Y $ is given by:

$$
Y = \Gamma^T (X - \mu)
$$

When you take the expectation of $ Y $, you apply the expectation operator to the transformation:

$$
\text{E}(Y) = \text{E}(\Gamma^T (X - \mu))
$$

Due to the linearity of the expectation operator, you can distribute it inside the transformation:

$$
\text{E}(Y) = \Gamma^T \text{E}(X - \mu)
$$

This works because $ \Gamma^T $ is a matrix of constants (not random), and $ \mu $ is the mean vector of $ X $, which is also constant (not random). Therefore, you can factor out the constant matrix $ \Gamma^T $ from the expectation, and since $ \mu $ is constant, $ \text{E}(\mu) = \mu $.

The expectation operator can go inside the transformation due to its linearity, and this is why you can write $ \text{E}(Y) $ as $ \Gamma^T \text{E}(X - \mu) $.

$\mu$ is the expected value of X hence ${E}(X - \mu) $ is 0.
```

---
```{admonition} Proof: $    \text{var}(Y) = \Gamma^T \Sigma \Gamma $
:class: dropdown

The variance of the transformed variables $ Y $ in PCA is related to the variance of the original variables $ X $ by the transformation matrix $ \Gamma $, which consists of the eigenvectors of the covariance matrix of $ X $.

Here's the step-by-step explanation of why the variance of $ Y $ is $ \Gamma^T \text{var}(X - \mu) \Gamma $:

1. **Transformation to Principal Components**:
   The principal components $ Y $ are obtained by applying the transformation matrix $ \Gamma $ to the centered data $ X - \mu $, where $ \mu $ is the mean of $ X $. The transformation is given by:

   $$
   Y = \Gamma^T (X - \mu)
   $$

2. **Variance of Transformed Variables**:
   By definition, the variance of the transformed variables $ Y $ is:

   $$
   \text{var}(Y) = E[(Y - E(Y))(Y - E(Y))^T]
   $$

   Because $ E(Y) = \Gamma^T E(X - \mu) = \Gamma^T \cdot 0 = 0 $ (as the data has been centered), this simplifies to:

   $$
   \text{var}(Y) = E[YY^T]
   $$

3. **Application of Transformation**:
   Substituting $ Y $ with the transformation $ \Gamma^T (X - \mu) $, we get:

   $$
   \text{var}(Y) = E[(\Gamma^T (X - \mu))(\Gamma^T (X - \mu))^T]
   $$

4. **Matrix Multiplication**:
   When you multiply out the matrices, you use the property that $ (AB)^T = B^T A^T $ for matrix transpose:

   $$
   \text{var}(Y) = E[\Gamma^T (X - \mu)(X - \mu)^T \Gamma]
   $$

5. **Linearity of Expectation**:
   Because expectation is a linear operator, you can move it inside the expression:

   $$
   \text{var}(Y) = \Gamma^T E[(X - \mu)(X - \mu)^T] \Gamma
   $$

6. **Covariance Matrix of $ X $**:
   The term $ E[(X - \mu)(X - \mu)^T] $ is the covariance matrix of $ X $, denoted by $ \Sigma $ or $ \text{var}(X) $:

   $$
   \text{var}(Y) = \Gamma^T \Sigma \Gamma
   $$
```

> Remember that our goal is to project p-dimensional data on just a few dimensions so that we can visualize them more easily. Thus in practice we often take q much smaller than p if p is large.

