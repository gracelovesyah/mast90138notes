
# Week 6
Pages: 163 - 193

## PCR (Principal Component Regression)

In PCR, we perform regression on the dependent variable, $ Y $, represented as follows:

### Regression Model

$$ Z = \alpha + \beta^T x + \epsilon $$

Where $ Z $ is the dependent variable, $ \alpha $ is the intercept, $ \beta $ is the coefficient vector, $ x $ is the vector of predictors, and $ \epsilon $ is the error term.

### Estimation of Coefficients
To estimate the coefficients $ \beta $, we minimize the sum of squared errors:

$$ \hat{\beta} = \arg \min \sum (Z_i - \beta^T x_i)^2 $$

The solution is given by:

$$ \hat{\beta} = (X^T X)^{-1} X^T Z $$

Given that $ X = \Gamma Y $, where $ \Gamma $ is the matrix of principal components, the PCR estimate of $ \beta $ becomes:

$$ \hat{\beta}_{\text{PC}} = (Y^T Y)^{-1} Y^T Z $$

## PLS (Partial Least Squares)

In PLS, regression is performed on a transformed set of predictors, $ T $.

### Transformation of Predictors
The transformation is defined as:

$$ T = \Phi X $$

Where $ \Phi $ is a matrix with $ |\Phi| = 1 $, ensuring normalization, and the covariance between $ Z $ and $ X $ is maximized.

### Estimation of Coefficients
The PLS estimate of $ \beta $ is found using:

$$ \hat{\beta}_{\text{PLS}} = (T^T T)^{-1} T^T Z $$

In this formulation, $ T $ represents the PLS components, which are linear combinations of the original predictors $ X $, optimized to explain the variance in $ Z $.

