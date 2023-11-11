# week6 lec1

## Partial least squares

Partial Least Squares (PLS) regression is indeed a method often compared to Principal Component Regression (PCR). Both are used for reducing the dimensionality of predictors, but while PCR only focuses on explaining the variance in the predictor variables $X$, PLS also takes into account the variance in the response variable $Z$.

Here's a breakdown of how PLS works and its main idea:

1. **Projection of $X$ and $Z$**:
    - PLS seeks to find the linear combination of the predictors (columns of $X$) such that, when we project both $X$ and $Z$ onto this new axis, the covariance between the projections is maximized.
    - This projection for $X$ results in components $T_1, T_2, ... , T_p$.

2. **Deflation**:
    - After finding the first set of projections, the variance that has been accounted for is removed or "deflated" from both $X$ and $Z$.
    - This deflation ensures that subsequent components are orthogonal to previous ones, similar to the process in PCA.

3. **Iterative Process**:
    - Steps 1 and 2 are repeated iteratively to find subsequent components, each of which aims to maximize the covariance between the projections of $X$ and $Z$.

4. **Regression**:
    - Once a desired number of components are obtained (usually far fewer than the original number of predictors), a regression model is fitted using these components as predictors to predict $Z$.

**Benefits of PLS**:
- By focusing on both the variance in $X$ and the covariance between $X$ and $Z$, PLS often results in components that are more relevant for predicting the response variable than PCR.
- Like PCR, PLS can handle multicollinearity by reducing the predictor variables to a smaller set of uncorrelated components.
- Useful for situations where the number of predictors $p$ is greater than the number of observations $n$ (i.e., $p > n$).

**Drawbacks of PLS**:
- While the components can sometimes be interpreted in the context of the original predictors, they are generally less interpretable than the original variables.
- The determination of the number of components to retain can be somewhat subjective and usually requires methods like cross-validation.

In essence, the key distinction between PCR and PLS is the objective. PCR seeks to capture the most variance in the predictors, regardless of the response. PLS, on the other hand, seeks to capture the most variance in the predictors that is also relevant to the response. As such, PLS tends to be more appropriate when prediction of the response is the main goal.

---
