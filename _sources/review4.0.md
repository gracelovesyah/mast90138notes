# review4

## QDA cont.

```{admonition} Describe the underlying model of quadratic discriminant analysis (QDA) for a general K-class problem (K > 2), and write down its decision rules (7)
:class: dropdown

Quadratic Discriminant Analysis (QDA) is a statistical method used for classification, and it is an extension of Linear Discriminant Analysis (LDA). While LDA assumes that different classes share the same covariance matrix, QDA allows each class to have its own covariance matrix. This makes QDA more flexible than LDA, particularly when classes have different shapes in the feature space.

### Underlying Model for K-Class Problem

In a K-class problem, we have $ K $ different classes. The underlying model for QDA assumes the following:

1. **Probability Distributions**: Each class $ k $ is modeled as a multivariate normal distribution with its own mean vector $ \mu_k $ and covariance matrix $ \Sigma_k $.

2. **Prior Probabilities**: Each class has a prior probability $ \pi_k $ which represents the probability of a randomly chosen observation belonging to class $ k $.

3. **Decision Boundary**: The decision boundaries between classes are quadratic, which is where QDA gets its name. This is due to the fact that each class has its own covariance matrix.

### Decision Rules

The decision rule in QDA is to assign an observation $ x $ to the class for which the quadratic discriminant function has the highest value. The quadratic discriminant function for class $ k $ is given by:

$$
\delta_k(x) = -\frac{1}{2} \ln|\Sigma_k| - \frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) + \ln \pi_k
$$

Where:

- $ |\Sigma_k| $ is the determinant of the covariance matrix for class $ k $.
- $ \Sigma_k^{-1} $ is the inverse of the covariance matrix for class $ k $.
- $ \mu_k $ is the mean vector for class $ k $.
- $ \pi_k $ is the prior probability of class $ k $.

The observation $ x $ is assigned to the class $ k $ that maximizes $ \delta_k(x) $.

### Summary

In summary, QDA is used in situations where the assumption of equal covariance matrices for all classes (as in LDA) is not tenable. By allowing each class to have its own covariance matrix, QDA can model more complex class distributions, at the cost of needing more parameters to be estimated, which can lead to overfitting if the sample size is not large enough.

---

The quadratic discriminant function $\delta_k(x)$ in Quadratic Discriminant Analysis (QDA) is derived from the likelihood of the data given a particular class, under the assumption that the data in each class follow a multivariate normal (Gaussian) distribution. Here's a step-by-step explanation of how $\delta_k(x)$ is obtained:

### Starting Point: Multivariate Normal Distribution

For a K-class problem, each class $ k $ is assumed to follow a multivariate normal distribution with its own mean vector $ \mu_k $ and covariance matrix $ \Sigma_k $. The probability density function of a multivariate normal distribution is given by:

$$
f(x | \mu_k, \Sigma_k) = \frac{1}{(2\pi)^{n/2} |\Sigma_k|^{1/2}} \exp \left( -\frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) \right)
$$

Where:

- $ x $ is the data point.
- $ \mu_k $ is the mean vector for class $ k $.
- $ \Sigma_k $ is the covariance matrix for class $ k $.
- $ |\Sigma_k| $ is the determinant of $ \Sigma_k $.
- $ \Sigma_k^{-1} $ is the inverse of $ \Sigma_k $.

### Incorporating Prior Probabilities

In QDA, we also consider the prior probabilities of each class $ \pi_k $. The overall likelihood of a data point $ x $ belonging to class $ k $ is the product of its class-conditional density and the class prior probability:

$$
P(k | x) \propto \pi_k f(x | \mu_k, \Sigma_k)
$$

### Taking the Logarithm

To simplify calculations and avoid numerical underflow, we take the logarithm of this expression, leading to the log-likelihood:

$$
\ln P(k | x) = \ln \pi_k + \ln f(x | \mu_k, \Sigma_k)
$$

Plugging in the expression for $ f(x | \mu_k, \Sigma_k) $ and simplifying, we get:

$$
\ln P(k | x) = \ln \pi_k - \frac{1}{2} \ln |\Sigma_k| - \frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) - \text{constant}
$$

Since the constant term (involving $(2\pi)^{n/2}$) is the same for all classes and does not affect the classification decision, it can be omitted. This gives us the quadratic discriminant function for class $ k $:

$$
\delta_k(x) = \ln \pi_k - \frac{1}{2} \ln |\Sigma_k| - \frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k)
$$

### Decision Rule

Finally, the decision rule in QDA is to assign the observation $ x $ to the class that maximizes $\delta_k(x)$, which is equivalent to maximizing the log-likelihood $ \ln P(k | x) $. This accounts for both the shape of the class distribution (through the covariance matrix) and the prior probability of each class.

```