# week7 lec1
## PCA vs PLS
Here's a simplified comparison of PCA and PLS in a table format:

| Aspect                  | PCA                                | PLS                                            |
|-------------------------|------------------------------------|------------------------------------------------|
| **Purpose**             | Dimensionality reduction           | Dimensionality reduction & Regression          |
| **Method**              | Unsupervised                       | Supervised                                     |
| **Component Extraction**| Maximizes variance of data         | Maximizes covariance between predictors & response |
| **Use Cases**           | Data visualization, exploratory data analysis | Predictive modeling with collinear predictors |
| **Outputs**             | Orthogonal principal components    | PLS components related to outcome prediction   |
| **Assumptions**         | Maximum variance is informative    | Variance shared with the response is informative |
| **Limitations**         | May not capture relevant patterns for prediction | Might miss informative variance not shared with response |
| **Interpretation**      | Typically lacks clear physical meaning | Somewhat more interpretable, but still complex |

---


## 6 Factor Analysis

Factor Analysis (FA) is a statistical method that seeks to uncover hidden factors or latent structures behind observed data. In essence, it aims to identify the underlying reasons for the correlations present among data points.

The Factor Analysis model can be expressed as:

$$ X = \Gamma Y + \mu = QF + \mu $$

In this model, $ Q $ is the Loading Matrix, which contains the loadings of observed variables on the factors, and $ F $ is the Factor Scores Matrix, representing the scores of each observation on the latent factors.

The Loading Matrix $ Q $ is derived from:

$$ Q = \Gamma \Lambda^{1/2} $$

And the Factor Scores Matrix $ F $ is obtained by:

$$ F = \Lambda^{1/2} Y $$

The goal of FA is to explain $ p $ variables using $ q $ factors, where $ q < p $.

However, FA cannot always be directly represented as $ X = QF + \mu $. We often need to consider a specific factor, $ U $ (unique factors), which is an error term matrix. This matrix accounts for the random errors that cannot be explained by the relationship between latent factors and observed variables. It includes the random noise in the data and the variability not captured by the factor analysis model.

It is important to note that the expectation of $ U $ is zero, $ E(U) = 0 $, and that the covariance between any two specific factors $ U_i $ and $ U_j $ is zero, $ \text{COV}(U_i, U_j) = 0 $, as well as the covariance between the factor scores $ F $ and specific factors $ U $, $ \text{COV}(F, U) = 0 $. This means that the errors $ U $ are uncorrelated with each other, and also uncorrelated with the factor scores $ F $.

--- 

## 6.1 ORTHOGONAL FACTOR MODEL

## 6.2 INTERPRETING THE FACTORS

## 6.3 SCALING THE DATA
