# review3

## Classification Models (Binary)
Understood, here's a simplified and correctly aligned version of the classification models:

### Linear Regression (for Binary Classification)

The probability model for class $ G = 1 $ is given by:

$$
m(x_1) = P(G = 1|X=x) = \beta_0 + \beta^T X
$$

For class $ G = 2 $, the probability model is the complement of $ m(x_1) $:

$$
m(x_2) = 1 - m(x_1) = 1 - (\beta_0 + \beta^T X)
$$

The decision boundary where we classify $ X $ to $ G = 1 $ if:

$$
\beta_0 + \beta^T X > 1/2
$$

### Logistic Regression (for Binary Classification)

The probability model for class $ G = 1 $ is given by:

$$
m(x_1) = P(G = 1|X=x) = \frac{e^{\beta_0 + \beta^T X}}{1+e^{\beta_0 + \beta^T X}}
$$

For class $ G = 2 $, the probability model is the complement of $ m(x_1) $:

$$
m(x_2) = 1 - m(x_1) = \frac{1}{1+e^{\beta_0 + \beta^T X}}
$$

The decision boundary where we classify $ X $ to $ G = 1 $ if:

$$
\beta_0 + \beta^T X > 0
$$

The last inequality comes from the fact that the logistic function yields a probability greater than 0.5 when $ \beta_0 + \beta^T X > 0 $, which is the condition for classifying $ X $ into $ G = 1 $ for logistic regression.


## Multivariate Normal (Gaussian)

For a univariate normal distribution, the probability density function (pdf) is given by:

$$ f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) $$

where $ \mu $ is the mean and $ \sigma^2 $ is the variance of the distribution.

For a multivariate normal distribution, the pdf becomes:

$$ f(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^k|\Sigma|}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right) $$

where:
- $ \mathbf{x} $ is a k-dimensional random vector.
- $ \boldsymbol{\mu} $ is the mean vector.
- $ \Sigma $ is the covariance matrix, which is a $ k \times k $ matrix.
- $ |\Sigma| $ is the determinant of the covariance matrix.
- $ \Sigma^{-1} $ is the inverse of the covariance matrix.

The term $ (\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}) $ is a quadratic form which generalizes the squared difference term $ (x-\mu)^2 $ from the univariate case. Here's why the matrix form is used:

1. **Covariance**: In multiple dimensions, not only do we have variances along each dimension (akin to $ \sigma^2 $ in the univariate case), but we also have covariances between each pair of dimensions. The covariance matrix $ \Sigma $ encapsulates all this information.

2. **Quadratic Form**: The quadratic form $ (\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}) $ measures the squared distance of $ \mathbf{x} $ from the mean $ \boldsymbol{\mu} $, weighted by the covariance structure. The transpose $ (\mathbf{x}-\boldsymbol{\mu})^T $ and the inverse covariance matrix $ \Sigma^{-1} $ are used to properly account for the orientation and scaling dictated by the covariance between the variables.

3. **Determinant and Inverse**: The determinant $ |\Sigma| $ serves to normalize the distribution so that the total probability integrates to 1. The inverse $ \Sigma^{-1} $ is used in the quadratic form to account for the correlation structure between the variables.

```{admonition} Assumption
1. LDA assumes that both classes share the same covariance matrix, which results in a linear decision boundary. 
2. Normal distribution.
```

### Linear Discriminant Analysis (for Binary Classification)

**Bayes**

$$

P(G = k |X = x) = \frac{P(X = x|G = k )\pi_k}{P(X=x)}
$$

where $\pi_k = P(G = k)$.


The probability model for class $ G = 1 $ is given by a Gaussian distribution with mean $ \mu_1 $ and common covariance matrix $ \Sigma $:

$$
m(x_1) = P(G = 1|X=x) \propto \exp\left(-\frac{1}{2}(x - \mu_1)^T \Sigma^{-1}(x - \mu_1)\right)
$$

Similarly, for class $ G = 2 $, the probability model is a Gaussian distribution with mean $ \mu_2 $ and the same common covariance matrix $ \Sigma $:

$$
m(x_2) = P(G = 2|X=x) \propto \exp\left(-\frac{1}{2}(x - \mu_2)^T \Sigma^{-1}(x - \mu_2)\right)
$$

The decision boundary for LDA is derived by setting the log of the ratio of these probabilities equal to zero, which simplifies to a linear function in $ x $ due to the common covariance matrix:

$$
\log\left(\frac{m(x_1)}{m(x_2)}\right) = 0
$$

This simplifies further to the linear equation:

$$
(x - \mu_1)^T \Sigma^{-1}(x - \mu_1) - (x - \mu_2)^T \Sigma^{-1}(x - \mu_2) = 0
$$

Which can be simplified to:

$$
x^T \Sigma^{-1}(\mu_1 - \mu_2) - \frac{1}{2}(\mu_1 + \mu_2)^T \Sigma^{-1}(\mu_1 - \mu_2) + \log\left(\frac{\pi_1}{\pi_2}\right) = 0
$$

Here, $ \mu_1 $ and $ \mu_2 $ are the mean vectors for each class, $ \Sigma $ is the shared covariance matrix, and $ \pi_1 $ and $ \pi_2 $ are the prior probabilities of each class.

The resulting decision boundary is a hyperplane in the feature space, and the side of the hyperplane an observation falls on determines the predicted class for that observation. The term $ x^T \Sigma^{-1}(\mu_1 - \mu_2) $ represents the projection of the difference in class means onto the data, and it is this projection that is used for classification.

```{tip}
What if data is non-linearly separable?
Use QD (instead of LD)! 
```
### Quadratic Discriminant Analysis (for Binary Classification)

In QDA, each class $ G = 1 $ and $ G = 2 $ has its own covariance matrix, which allows for the modeling of more complex decision boundaries.

The probability model for class $ G = 1 $ is given by:

$$
m(x_1) = P(G = 1|X=x) \propto \exp\left(-\frac{1}{2}(x - \mu_1)^T \Sigma_1^{-1}(x - \mu_1)\right)
$$

For class $ G = 2 $, the probability model is:

$$
m(x_2) = P(G = 2|X=x) \propto \exp\left(-\frac{1}{2}(x - \mu_2)^T \Sigma_2^{-1}(x - \mu_2)\right)
$$

The decision boundary is found by setting $ m(x_1) = m(x_2) $ and solving for $ x $. In practice, this means finding the $ x $ such that:

$$
-\frac{1}{2}(x - \mu_1)^T \Sigma_1^{-1}(x - \mu_1) + \log(\pi_1) = -\frac{1}{2}(x - \mu_2)^T \Sigma_2^{-1}(x - \mu_2) + \log(\pi_2)
$$

### Regularization

Regularized Quadratic Discriminant Analysis (QDA) is an extension of QDA that addresses the potential problem of overfitting, especially in situations where the number of features is large compared to the number of observations, or when the class covariance matrices are nearly singular.

In regular QDA, you estimate the covariance matrices for each class separately, which can lead to models that are very flexible but also prone to overfitting the data. Regularization introduces a shrinkage parameter, which essentially pulls the class-specific covariance matrices towards a common covariance matrix or towards the identity matrix (which would imply independence of the features).

Here's a bit more detail on how it works:

1. **Covariance Estimation**: In standard QDA, the covariance matrix for each class $ \Sigma_k $ is estimated directly from the data corresponding to that class.

2. **Regularization**: In regularized QDA, you adjust each $ \Sigma_k $ by pulling it towards a central estimate $ \Sigma $ or towards the identity matrix $ I $. The extent to which you do this is controlled by a regularization parameter $ \lambda $ (which can be between 0 and 1).

3. **Regularized Covariance Matrix**: The regularized covariance matrix for each class $ k $ can be calculated as:

   $$
   \hat{\Sigma}_k = (1 - \lambda) \Sigma_k + \lambda \Sigma
   $$

   where $ \Sigma $ could be the average of all the $ \Sigma_k $ (the pooled covariance matrix), or it could be the identity matrix if you're regularizing towards independence.

4. **Logarithmic Loss with Regularization**: The regularization modifies the discriminant function by changing the covariance matrices used in the function. The regularized discriminant function incorporates the regularized covariance matrices to compute the probability that an observation belongs to each class.

5. **Choosing $ \lambda $**: The regularization parameter $ \lambda $ is typically chosen through cross-validation to optimize the model's predictive performance on unseen data.

Regularization helps to improve the model's generalizability and robustness by preventing it from being overly complex relative to the amount of training data. It is particularly useful when the data contains redundant features or when there is collinearity among the features.

--- 

- [Partial least squares regression (PLSR) - explained](https://www.youtube.com/watch?v=SWfucxnOF8c)

- Similar to principal component regression, the method of partial least squares, also called projection to latent structures, is a technique that reduces the number of explanatory variables to a smaller set of uncorrelated variables.

- Both methods are used to overcome the problem with collinearity in linear regression and in cases where we have more explanatory variables than observations. Watch the lecture about principal component regression for more information.