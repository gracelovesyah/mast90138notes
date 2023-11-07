# review2

## PCA Cont

```{image} ./images/pca1.png
:alt: pca
:class: bg-primary mb-1
:width: 800px
:align: center
```

- the reason we centralise data: so that calculating the variance of projected dots on pc is equivalent to calculating the sum of the distance of each dot to the center (0,0) => more interpretable and simplify calculation process.


```{image} ./images/pca2.png
:alt: pca
:class: bg-primary mb-1
:width: 800px
:align: center
```

 singular value decomposition (SVD): to "recover" the point we draw.

## Eigenvalue decomposition

Eigenvalue decomposition is a technique in linear algebra where a matrix is broken down into its constituent parts to make certain operations on the matrix easier to perform. This decomposition is especially useful in the context of covariance matrices when analyzing data.

Let's go step by step to understand how the eigenvalue decomposition of a covariance matrix works and why it's useful.

### Covariance Matrix

Firstly, the covariance matrix is a square matrix that summarizes the covariance (a measure of how much two variables change together) between each pair of elements in a data set. If you have a data set with \( n \) dimensions, the covariance matrix will be \( n \times n \).

### Eigenvalues and Eigenvectors

An eigenvector of a square matrix \( A \) is a non-zero vector \( v \) such that when \( A \) is multiplied by \( v \), the product is a scalar multiple of \( v \). That scalar is known as an eigenvalue. Mathematically, this is represented as:

\[
A v = \lambda v
\]

where \( A \) is the matrix, \( v \) is the eigenvector, and \( \lambda \) is the eigenvalue.

### Decomposition of a Covariance Matrix

When you decompose a covariance matrix \( \Sigma \), you find its eigenvalues and eigenvectors. This decomposition has the form:

\[
\Sigma = Q \Lambda Q^{-1}
\]

where \( \Sigma \) is the covariance matrix, \( Q \) is the matrix composed of the eigenvectors of \( \Sigma \), \( \Lambda \) is the diagonal matrix with the eigenvalues of \( \Sigma \) on the diagonal, and \( Q^{-1} \) is the inverse of the matrix \( Q \).

### Why Decompose a Covariance Matrix?

Decomposing a covariance matrix is useful for several reasons:

1. **Principal Component Analysis (PCA):** PCA is a technique to reduce the dimensionality of data. It identifies the directions (principal components) in which the data varies the most. In PCA, the eigenvectors (principal components) of the covariance matrix provide the directions of maximum variance, and the eigenvalues indicate the magnitude of that variance.

2. **Efficiency:** Once decomposed, certain operations such as matrix inversion or determining the matrix rank become much easier and more computationally efficient.

3. **Understanding Data:** By examining the eigenvalues and eigenvectors, one can understand the shape and distribution of the data. Large eigenvalues correspond to dimensions with large variance, indicating that the data spreads out widely in the direction of the corresponding eigenvector.

Would you like to see a numerical example of how to perform eigenvalue decomposition on a covariance matrix?