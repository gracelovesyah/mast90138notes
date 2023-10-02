# basics

## Names and defs

The abbreviations and symbols you mentioned are commonly associated with statistical methodologies like Principal Component Analysis (PCA) and Principal Component Regression (PCR). Let's break down the meanings:

### **PCA (Principal Component Analysis):**
   PCA is a dimensionality reduction technique that's often used to reduce the number of variables in a dataset while retaining most of the original variability. It does this by projecting the original data into a new coordinate system defined by the principal components.

### **$X$**:
 This typically represents the original data matrix with rows as observations (e.g., individuals) and columns as variables (e.g., features or measurements). Each entry $X_{ij}$ represents the measurement of the $j^{th}$ variable for the $i^{th}$ observation.
  
### **$\Gamma$: (Gamma)**
The columns of $\Gamma$ are the eigenvectors of the covariance matrix of $X$. Each column (eigenvector) points in the direction of a principal component. These eigenvectors are also often referred to as "loadings."

- In other words, $\Gamma$ provides the weights or coefficients that transform the original data $X$ into its principal components $Y$. If $X$ has $p$ predictors, then $\Gamma$ will be a $p \times p$ matrix, where the first column is the loading vector for PC1, the second column is the loading vector for PC2, and so on.

The matrix of eigenvectors, denoted as $\Gamma$, contains the principal directions of the dataset. Each column of $\Gamma$ represents an eigenvector, ordered by the corresponding eigenvalue in descending order.


$$
\Gamma = \begin{bmatrix}
a_{11} & a_{12} & \dots & a_{1p} \\
a_{21} & a_{22} & \dots & a_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
a_{p1} & a_{p2} & \dots & a_{pp} \\
\end{bmatrix}
$$

### **$Y$:**
When you project the original data $X$ onto the principal components (or eigenvectors), you get a new data representation, $Y$. Here, $Y$ contains the scores of the observations on the principal components. The first column of $Y$ contains the scores for PC1, the second column has the scores for PC2, and so on.

$$ Y = Γ^T X $$

### **$Z$**: 
This represents the predicted outcome variable, based on the regression model. 

$$ Z = \beta^T X $$

- In the equation $Z = \beta^T X + \epsilon$, $Z$ is the predicted value, $\beta^T$ is the transpose of the coefficient vector, and $\epsilon$ is the error term.
### **$\Lambda$ (Lambda)**: 

This is a diagonal matrix of the variances of the principal components. In PCA, the components are ordered by the amount of variance they explain in the original data. The first principal component explains the most variance, the second explains the next highest amount, and so on. These variances are the eigenvalues of the covariance matrix of $X$.



$$
\Lambda = \begin{bmatrix}
\lambda_1 & 0 & \cdots & 0 \\
0 & \lambda_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \lambda_n \\
\end{bmatrix}
$$

### $ \lambda_1 $


The symbol $ \lambda_1 $ typically refers to the first (and largest) eigenvalue when discussing matrices, especially in the context of Principal Component Analysis (PCA).

In PCA, $ \lambda_1 $ represents the amount of variance explained by the first principal component. The eigenvalues (like $ \lambda_1 $) are obtained from the covariance (or sometimes correlation) matrix of the dataset.

Here's a brief overview of how $ \lambda_1 $ is calculated:

1. **Standardize the Data (if needed):** Before PCA, it's common to standardize each variable so they all have a mean of 0 and standard deviation of 1. This ensures that all variables are on a comparable scale.

2. **Compute the Covariance Matrix:** If $ X $ is the centered data matrix, the sample covariance matrix $ S $ is given by:

   $$ S = \frac{1}{n-1} X^T X $$

   where $ n $ is the number of observations.

3. **Compute Eigenvalues and Eigenvectors of the Covariance Matrix:** For the covariance matrix $ S $, we want to find scalars $ \lambda $ (eigenvalues) and vectors $ v $ (eigenvectors) such that:

   $$ S v = \lambda v $$

   There are many algorithms and software packages (like in Python or R) that can compute the eigenvalues and eigenvectors for you.

4. **Order Eigenvalues and Eigenvectors:** Once you have all the eigenvalues, you'll order them from largest to smallest. $ \lambda_1 $ would be the largest eigenvalue, $ \lambda_2 $ would be the second largest, and so on. The eigenvector associated with $ \lambda_1 $ is the first principal component.

As a note, $ \lambda_1 $ (and other eigenvalues) can be interpreted as the amount of variance explained by the corresponding principal component. The ratio $ \lambda_1 $ divided by the sum of all eigenvalues gives the proportion of total variance explained by the first principal component.


---

### **PCR (Principal Component Regression):**
### Z
$$ Z = m_{pc}(Y_1) + ϵ $$
$$ Z = β^T Γ_1Y_1 + ϵ $$



### Example
Of course! Let's use a very simple dataset as an example.

**Data:**
Consider we have three data points in 2-dimensional space:
```
X1 = [1, 2]
X2 = [2, 3]
X3 = [3, 5]
```
We will treat these as column vectors, and let's stack them into a matrix:
```
      [1 2 3]
X =   [2 3 5]
```

**Step 1: Center the Data**
First, we'll center the data by subtracting the mean of each variable.
```
Mean of first variable (rows): (1 + 2 + 3) / 3 = 2
Mean of second variable (columns): (2 + 3 + 5) / 3 = 10/3

Centered data:
      [-1  0  1]
X' =  [-2/3 1/3 5/3]
```

**Step 2: Compute the Covariance Matrix**
The covariance matrix \( S \) for \( X' \) is:
```
      sum(x1i * x1i)      sum(x1i * x2i)
S =   sum(x1i * x2i)      sum(x2i * x2i)
```
For the small data set:
```
      [2/3  4/3]
S =   [4/3  14/3]
```

**Step 3: Calculate Eigenvectors and Eigenvalues of the Covariance Matrix**
This is the most involved step. Here, we're looking for vectors $ \Gamma $ and scalars $ \lambda $ such that $ S\Gamma = \lambda\Gamma $. 

For our simple example, you'd typically use software or a mathematical method to find the eigenvectors and eigenvalues. Let's assume (for the sake of simplicity) that:
```
Eigenvector 1 (associated with the largest eigenvalue λ1): Γ1 = [a, b]
Eigenvector 2 (associated with the smaller eigenvalue λ2): Γ2 = [c, d]
```

**Step 4: Project Data onto Eigenvectors (Principal Components)**
To get the projection of the data onto the principal components, you multiply the transposed eigenvector matrix with the centered data.
```
      [-1   0   1]       [a c]
Y =   [-2/3 1/3 5/3]  *  [b d]
```
The resulting matrix $ Y $ gives the scores of each original data point on the principal components.

In real scenarios with larger datasets and higher dimensions, you'd typically use software packages like Python's `scikit-learn` or R's built-in functions to do PCA. They handle the numerical details and optimizations behind the scenes.

### Communality
In multivariate statistics, particularly in factor analysis, "communality" refers to the portion of the variance of a given variable that is accounted for by the common factors. In simpler terms, it's the variance in a particular variable explained by the factors in a factor analysis.

Here's a more detailed breakdown:

1. **Total Variance**: For each variable in a dataset, there's a total amount of variance associated with that variable.

2. **Unique Variance**: A part of this total variance is unique to the variable, meaning it's not shared with any other variables in the analysis. This unique variance could be due to unique factors or error.

3. **Common Variance or Communality**: The remaining variance (i.e., total variance minus unique variance) is the variance shared with other variables. This shared variance is what the factors in factor analysis aim to represent. 

Mathematically, for a given variable:
$
\text{Total Variance} = \text{Unique Variance} + \text{Communality}
$

In the context of factor analysis:

- High communality for a variable indicates that a large portion of its variance is accounted for by the factors.
  
- Low communality indicates that the factors do not explain a significant portion of the variance of that variable.

When you conduct factor analysis, you're essentially trying to find underlying factors that account for the communalities among variables, which helps in reducing dimensionality and in understanding the underlying structure of the data.

```{admonition} Communality Example
:class: tip, dropdown

Suppose you are conducting a study on student performance, and you have three measures:

1. Math scores
2. Physics scores
3. Writing scores

You suspect there are underlying factors (like numerical aptitude and verbal aptitude) that might explain correlations among these scores.

After conducting factor analysis on the data, you extract the communalities for each measure:

1. Math scores: 0.85
2. Physics scores: 0.82
3. Writing scores: 0.70

Here's how to interpret these communalities:

1. **Math scores (0.85)**: 85% of the variance in math scores can be explained by the extracted factors. Given that Physics also has a high communality, it's likely that there's a common factor like "Numerical Aptitude" that both Math and Physics scores relate to.

2. **Physics scores (0.82)**: Similarly, 82% of the variance in physics scores can be explained by the extracted factors. Again, this could be due to the "Numerical Aptitude" factor.

3. **Writing scores (0.70)**: 70% of the variance in writing scores can be explained by the extracted factors. This might indicate a different factor, like "Verbal Aptitude", influencing Writing scores. The fact that it's slightly lower might also indicate that there are other specific skills or knowledge unique to Writing (not captured by the factors) that account for the remaining 30% of the variance.

This is a simplified example, but it demonstrates the concept. In practice, you'd use the factor loadings to further interpret and label the factors. The communalities give you an idea of how well each variable fits into the factor structure you've identified.
```