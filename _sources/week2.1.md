
# Week 2 Lecture 1

## 3. Review of Matrix Properties Continued
- **Quadratic Form**: 
  - A quadratic form in variables $ x_i $ with respect to matrix $ A $ is given by $ Q(x) = \sum a_{ij}x_i x_j = x^TAx $.
  - A matrix is **positive semidefinite** if $ x^TAx \geq 0 $ for all $ x $, and **positive definite** if $ x^TAx > 0 $ for all $ x \neq 0 $.

- **Euclidean Distance**:
  - The weighted version of Euclidean distance takes into account the different importances or scales of various dimensions.

- **Norm**: 
  - A norm is a function that assigns a strictly positive length or size to all vectors in a vector space, except for the zero vector, which is assigned a length of zero.

- **Angle Between Two Vectors**:
  - The cosine of the angle $ \theta $ between two vectors $ x $ and $ y $ can be computed using the dot product: 
    
$$ \cos(\theta) = \frac{x^T y}{\|x\|\|y\|} $$
.

- **Rotation**:
  - An **orthogonal matrix** represents a rotation and is defined as a square matrix whose columns and rows are orthogonal unit vectors.
  - The matrix $ \Gamma $ for rotation by an angle $ \theta $ is:
    
$$
    \Gamma = \begin{pmatrix}
    \cos(\theta) & \sin(\theta) \\
    -\sin(\theta) & \cos(\theta)
    \end{pmatrix}
    $$

  - When applying $ \Gamma $ to a vector $ x $, $ y = \Gamma x $ represents a counterclockwise rotation through the origin, while $ y = \Gamma^T x $ represents a clockwise rotation.

## 3. Mean, Covariance, Correlation
- The covariance $ \sigma_{XY} $ between two random variables $ X $ and $ Y $ quantifies the degree to which they linearly relate to each other:
  
$$ \sigma_{XY} = \text{cov}(X, Y) = E(XY) - E(X)E(Y) $$


- $ \Sigma $, the covariance matrix, has the following properties:
  - It is symmetric: $ \Sigma = \Sigma^T $.
  - It is semi-positive definite: $ \Sigma \geq 0 $.

- $ \Sigma $ is defined as $ \Sigma = E\{(X - \mu)(X - \mu)^T\} $, where $ \mu $ is the mean vector of the random variable $ X $.

- In practice, $ \Sigma $ can be estimated from an i.i.d sample $ X_1, \ldots, X_n $ using the sample covariance matrix $ S $, which shares the properties of symmetry and semi-positive definiteness.

- The sample covariance matrix $ S $ can be computed as:
  
$$ S = \frac{1}{n-1}\left(X^TX - \frac{1}{n}\bar{X}^T\bar{X}\right) $$

  where $ X $ is the data matrix and $ \bar{X} $ is the column vector of sample means.

- **Problem with Covariance**: It is not unit invariant. Changing the units of measurement changes the covariance values.

### Solution: Correlation
- The correlation coefficient $ \rho_{ij} $ between two variables is defined as:
  
$$ \rho_{ij} = \frac{\sigma_{ij}}{\sqrt{\sigma_{ii}\sigma_{jj}}} $$

  This coefficient always lies between -1 and 1.
  - $ |\rho_{ij}| = 1 $ indicates a perfect linear relationship.
  - $ \rho_{ij} = 0 $ indicates no linear relationship, but does not necessarily imply independence.

- The correlation matrix $ R $ can be computed using the formula:
  
$$ R = D^{-1/2}SD^{-1/2} $$

  where $ S $ is the sample covariance matrix, and $ D $ is the diagonal matrix of sample variances.
