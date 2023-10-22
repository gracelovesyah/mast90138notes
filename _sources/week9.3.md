# week9 additional notes


## LDA

- [StatQuest: Linear Discriminant Analysis (LDA) clearly explained.](https://www.youtube.com/watch?v=azXCzI57Yfc)

Linear Discriminant Analsys (LDA) is like PCA, but it focuses on maximzing the seperatibility among known categories.

```{image} ./images/lda1.png
:alt: 9.3
:class: bg-primary mb-1
:width: 800px
:align: center
```
```{image} ./images/lda2.png
:alt: 9.3
:class: bg-primary mb-1
:width: 800px
:align: center
```
```{image} ./images/lda3.png
:alt: 9.3
:class: bg-primary mb-1
:width: 800px
:align: center
```

```{image} ./images/lda4.png
:alt: 9.3
:class: bg-primary mb-1
:width: 800px
:align: center
```

**Similarities between PCA (Principal Component Analysis) and LDA (Linear Discriminant Analysis):**

1. **Dimensionality Reduction**: Both PCA and LDA are techniques used for dimensionality reduction. They transform the original features into a lower-dimensional space.

2. **Linear Transformation**: Both methods achieve the reduction by finding new axes (or linear combinations of original features) to project the data onto.

3. **Ordered Axes**: As mentioned in your notes:
   - PCA orders its new axes (principal components) based on the amount of variance they capture from the data, with PC1 capturing the most and subsequent components capturing less.
   - LDA orders its axes (linear discriminants) based on how well they separate different classes. LD1 separates the classes the most, and subsequent discriminants do so to a lesser degree.

4. **Eigenvectors and Eigenvalues**: Both PCA and LDA use eigenvectors and eigenvalues in their calculations. In PCA, they are derived from the data's covariance matrix. In LDA, they arise from the ratio of between-class to within-class scatter matrices.

5. **Interpretability**: In both PCA and LDA, one can investigate which original features (like genes, in a biological context) are most influential in defining the new axes. This helps in understanding the underlying structure and importance of original features.

6. **Visualization**: Both PCA and LDA are commonly used for visualization purposes. By reducing the dimensionality to 2 or 3 components/discriminants, data can be visualized in 2D or 3D plots.

7. **Linearity**: Both techniques assume linear relationships among features. They work best when this assumption holds.

However, it's essential to remember the key difference: PCA is unsupervised and focuses on maximizing variance irrespective of class labels, while LDA is supervised and aims to maximize the separability between known classes.

## QDA

- [Machine Learning 3.2 - Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA)](https://www.youtube.com/watch?v=IMfLXEOksGc)

```{image} ./images/qda1.png
:alt: 9.3
:class: bg-primary mb-1
:width: 800px
:align: center
```
```{image} ./images/qda2.png
:alt: 9.3
:class: bg-primary mb-1
:width: 800px
:align: center
```

Review:

- LDA and ODA assume distributions are Gaussian, estimate from data, and classify by maximum likelihood.
- LDA assumes same Gaussian. (linear decision boundaries)
- QDA allows different Gaussians. (quadratic decision boundaries)

---

## Lab
Your task today will be to apply the linear discriminant analysis (LDA) and quadratic discriminant analysis (QDA) classifiers seen in class to these data: 

(1) use the training data to construct the classifier; 

(2) apply the classifier to the test data; 

(3) Compute the classification error of the classifier on those test data