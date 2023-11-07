# week5 lec2
## Principal Component Regression (PCR)
Principal Component Regression (PCR) is a regression technique that combines the ideas of Principal Component Analysis (PCA) and multiple linear regression. It's particularly useful when you have multicollinearity in your data, which means that some of your predictor variables are correlated with one another.

Principal component regression is especially used to overcome the problem with collinearity in linear regression by combining explanatory variables to a smaller set of uncorrelated variables

Here's how PCR works step-by-step:

1. **Principal Component Analysis (PCA)**:
   - PCA is performed on the predictor variables. This results in a set of principal components (PCs) which are linear combinations of the original predictor variables.
   - These principal components are orthogonal to each other, meaning they are uncorrelated.
   - The first principal component accounts for the most variance in the original predictor variables, the second accounts for the second most, and so on.

2. **Selection of Principal Components**:
   - Not all principal components are necessarily useful for regression. Typically, the first few PCs capture the majority of the variability in the data.
   - The number of PCs to keep can be determined through various methods: the scree plot, a specified percentage of variance explained, cross-validation to minimize prediction error, etc.

3. **Regression**:
   - Instead of performing regression on the original predictor variables, we regress the dependent variable on the selected principal components.
   - Since the PCs are uncorrelated, this addresses the multicollinearity issue.

4. **Back-Transformation (if necessary)**:
   - If you're interested in understanding the relationship between the original predictor variables and the dependent variable (as opposed to just prediction), you'll need to back-transform the regression coefficients from the PC space to the original variable space.

**Benefits of PCR**:
- Handles multicollinearity by reducing the predictor variables to a smaller set of uncorrelated components.
- Can reduce the noise in the data if only the PCs capturing the most variance are used.

**Drawbacks of PCR**:
- The resulting regression coefficients (in the PC space) are not as interpretable as in standard linear regression because they are based on orthogonal transformations of the original predictors.
- Does not necessarily lead to a model that optimally predicts the response variable, because PCA only considers the predictor variables and not their relationship with the response variable. 

For problems where prediction accuracy is of utmost importance, techniques like Partial Least Squares (PLS) regression or Ridge/Lasso regression might be more appropriate, as they take into account both predictor variance and their relationship with the response variable.

## Useful Resource
- Youtube Video: [Principal component regression (PCR) - explained](https://www.youtube.com/watch?v=SWfucxnOF8c)


---

## 11.7 Revision Notes

### Notation

$$
Z = m(x) + \epsilon
$$

where
$m(x)=E(Z|X =x)=\beta^Tx$
and our goal is to find the $\beta$ s.t.

$$
\hat{\beta} = argmin \sum(Z_i - b^TX_i)^2 = (X^TX)^{-1}X^TZ
$$

```{note}
Why is dimension reduction useful?
```

$X^TX$ is not invertible when p > n (i.e. number of feature bigger than number of observed data). Hence we need to perform dimension reduction to guarantee that we can find the best estimator of $\beta$. However this may not be accurate:  if $\hat{\beta}$ is far from $\beta$ then all it does is bring us some misleading information about the relationship between X and Z. Hence a solution is PCR.