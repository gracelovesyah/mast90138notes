# Comparison

## LDA & QDA

| **Feature**             | **LDA**                                                                                   | **QDA**                                                                           |
|-------------------------|-------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| **Covariance Matrix**   | Assumes a common covariance matrix for all classes.                                       | Estimates a separate covariance matrix for each class.                            |
| **Complexity**          | Simpler model due to common covariance assumption. Less computationally intensive.        | More complex model; requires estimating more parameters. More computationally intensive. |
| **Overfitting Risk**    | Lower risk of overfitting, especially with small datasets like Iris.                      | Higher risk of overfitting, as it estimates more parameters.                      |
| **Dimensionality Reduction** | Effective in reducing dimensions while maintaining class separability.                       | Does not focus on dimensionality reduction.                                       |
| **Data Distribution Assumption** | Performs well if predictor distributions are approximately normal and similar across classes. | Can model more complex class distributions but needs more data to estimate these distributions effectively. |
| **Regularization**      | Implicit regularization due to common covariance assumption.                              | Lacks the same level of implicit regularization.                                  |
| **Performance with Small Sample Size** | Generally better performance on small datasets due to simpler model and regularization.    | May struggle with small sample sizes due to the need to estimate more parameters. |
| **Best Use Case**       | More suitable for datasets where classes have similar covariance structures.               | More suitable for datasets with significantly different covariance structures in each class and larger sample sizes. |

This table illustrates why LDA often performs better than QDA on the Iris dataset, largely due to its simplicity, lower overfitting risk, effective dimensionality reduction, and better suitability for smaller datasets with similar class covariance structures.

## K-means and K-medoids


| **Feature**                  | **K-means**                                                      | **K-medoids**                                                   |
|------------------------------|------------------------------------------------------------------|-----------------------------------------------------------------|
| **Centroid Representation**  | Uses the mean of the points in a cluster as the cluster center.  | Uses an actual point from the dataset as the cluster center (medoid). |
| **Sensitivity to Outliers**  | Sensitive to outliers, as means are easily influenced by extreme values. | Less sensitive to outliers since medoids are less affected by extreme values. |
| **Objective Function**       | Minimizes the sum of squared distances between points and their respective cluster centroids. | Minimizes the sum of dissimilarities between points and their respective medoids. |
| **Scalability**              | Generally more scalable to large datasets due to simpler calculations. | Less scalable as it involves more complex calculations (e.g., pairwise distances between all points). |
| **Suitability for Different Data Types** | Best suited for numerical data where mean is a meaningful measure. | Suitable for various types of data, including non-metric data, as it relies on general dissimilarity measures. |
| **Algorithm Complexity**     | Computationally faster and simpler, especially for large datasets. | Computationally more intensive, particularly for large datasets due to the need to compute and store all pairwise distances. |
| **Robustness**               | Less robust due to its sensitivity to outliers and initial centroid placement. | More robust to outliers and noise, but still sensitive to initial medoid placement. |
| **Result Interpretation**    | Cluster centers (means) may not correspond to actual data points, making interpretation less intuitive. | Cluster centers are actual data points (medoids), making interpretation more intuitive. |

K-means is generally preferred for its computational efficiency, especially with large and well-separated numerical datasets. K-medoids, on the other hand, offers advantages in terms of robustness and flexibility, being more suitable for datasets with outliers or non-numerical data types, though at the cost of increased computational requirements.

## RF and DT and RT


| **Feature** | **Decision Tree** | **Regression Tree** | **Random Forest** |
|-------------|-------------------|---------------------|-------------------|
| **Type** | A single tree structure used for classification or regression. | A type of Decision Tree specifically used for regression problems (predicting continuous values). | An ensemble of Decision Trees, used for both classification and regression. |
| **Complexity** | Relatively simple model. Complexity depends on depth and number of nodes. | Similar to Decision Tree; complexity varies with depth. Designed to handle continuous data and complex relationships. | More complex model due to being an ensemble of multiple trees. |
| **Overfitting** | Prone to overfitting, especially if the tree is deep. | Also prone to overfitting, similar to Decision Trees. | Less prone to overfitting due to averaging/majority voting across multiple trees. |
| **Interpretability** | Highly interpretable with clear decision paths. | Interpretable as it provides a clear regression model for decision paths. | Less interpretable due to complexity of multiple trees. |
| **Handling Non-Linearity** | Handles non-linear relationships well. | Specifically designed to handle non-linear relationships in regression. | Very effective in handling non-linear relationships due to multiple trees capturing various aspects of the data. |
| **Feature Importance** | Provides insights on feature importance. | Similar to Decision Trees, offers insights on feature importance for regression tasks. | Offers a more robust insight into feature importance by averaging across multiple trees. |
| **Performance** | Performance can vary; may struggle with very complex datasets. | Good for regression tasks, but performance can be affected by overfitting. | Generally high performance, especially in cases where individual trees have uncorrelated errors. |
| **Robustness** | Sensitive to changes in data and can suffer from variance. | Similar to Decision Trees, can be sensitive to variance in data. | Robust to noise and variance in data, thanks to averaging over many trees. |
| **Suitability** | Suitable for simple to moderately complex tasks. | Best for regression problems with moderate complexity. | Ideal for both classification and regression in complex scenarios with large datasets. |

- **Decision Trees** are versatile and can be used for both classification and regression, but they can easily overfit.
- **Regression Trees** are decision trees used specifically for regression tasks, predicting continuous values.
- **Random Forests** are an ensemble of decision trees, offering better performance and robustness, especially in complex scenarios, but with reduced interpretability.

## PCA and FA and LDA

Here's a comparison table for Principal Component Analysis (PCA), Factor Analysis (FA), and Linear Discriminant Analysis (LDA):

| **Feature** | **Principal Component Analysis (PCA)** | **Factor Analysis (FA)** | **Linear Discriminant Analysis (LDA)** |
|-------------|---------------------------------------|--------------------------|----------------------------------------|
| **Objective** | Reduces dimensionality by finding new uncorrelated variables (principal components) that maximize variance. | Seeks to explain observed correlations between variables using latent (unobserved) factors. | Maximizes class separability for classification purposes. Focuses on finding a linear combination of features that best separates different classes. |
| **Methodology** | Uses orthogonal transformation to convert possibly correlated features into linearly uncorrelated principal components. | Identifies underlying factors that explain the correlations among variables. Assumes that observed variables are linear combinations of factors plus error terms. | Finds linear discriminants based on the concept that classes are separable by finding the directions that maximize the distance between class means and minimize the variance within each class. |
| **Type of Analysis** | Unsupervised – does not consider class labels in the data. | Unsupervised – does not consider class labels. Primarily used for exploring data structure. | Supervised – uses class labels to find the optimal separation. |
| **Assumptions** | Assumes linear relationships among variables. | Assumes that observed variables have underlying latent factors and that these factors are linearly related to the variables. | Assumes that data is normally distributed, classes have identical covariance matrices, and the features are statistically independent. |
| **Output** | Principal components (new set of orthogonal features). | Factors (latent variables) and factor loadings (relationship of each variable to the underlying factor). | Linear discriminants used for classification. |
| **Use Cases** | Used for feature extraction, data compression, and exploratory data analysis. | Used for uncovering latent structure in data, especially in psychology, social sciences, and market research. | Used in classification problems to find a linear combination of features that best separates different classes. |
| **Interpretability** | Components are linear combinations of original features but can be less interpretable due to being a mixture of all features. | Factor loadings can be interpreted, but the factors themselves are abstract and not always directly interpretable. | The discriminant function is directly related to the class labels, making it interpretable in the context of classification. |
| **Robustness and Limitations** | Sensitive to scaling of data; works best with linear relationships. | Requires sufficient sample size and careful interpretation; sensitive to the choice of the number of factors. | Performance depends on the assumptions of normality and equal covariance; sensitive to outliers. |

- **PCA** is primarily used for dimensionality reduction and feature extraction without considering class labels.
- **FA** is used to identify latent factors explaining observed correlations, useful in psychometrics and other fields.
- **LDA** is a supervised technique used for classification, focusing on maximizing the separability between different classes.

---

## PCR and OLS

| **Aspect**                | **Principal Component Regression (PCR)**                                                                                     | **Ordinary Least Squares (OLS)**                                                                                                  |
|---------------------------|------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| **Handling Multicollinearity** | Excellent. PCR reduces multicollinearity by transforming the predictors into a set of uncorrelated principal components.      | Poor. OLS is sensitive to multicollinearity, which can inflate the variance of the coefficient estimates and make them unstable. |
| **Dimensionality Reduction**   | Good. PCR is useful for dimensionality reduction, as it summarizes the predictor variables with fewer principal components.  | None. OLS uses all original predictor variables and does not inherently reduce dimensionality.                                   |
| **Interpretability**           | Lower. The principal components are linear combinations of original variables, which can be less interpretable.              | Higher. OLS maintains the original variables, making the model more interpretable in terms of those variables.                   |
| **Model Complexity**           | Variable. The number of components used can be adjusted, offering flexibility in model complexity.                            | Fixed. OLS uses all variables, leading to potentially more complex models, especially with many predictors.                     |
| **Data Requirements**          | Higher. PCR requires sufficient data to accurately estimate the principal components.                                        | Lower. OLS can be applied even with a smaller dataset, although issues like multicollinearity may still arise.                  |
| **Prediction Accuracy**        | Potentially High. By reducing noise and multicollinearity, PCR can improve prediction accuracy.                              | Variable. OLS can be very accurate if the assumptions of linear regression are met, but can struggle with multicollinearity.    |
| **Model Selection**            | More Complex. Choosing the number of principal components to include adds an extra layer of model selection.                 | Simpler. OLS does not require this additional step, though variable selection might still be necessary.                         |
| **Computational Efficiency**   | Lower. Calculating principal components adds extra computational steps.                                                      | Higher. OLS is generally computationally simpler and faster, especially for smaller datasets.                                   |

In summary, PCR is particularly advantageous in situations with multicollinearity and when dealing with a large number of predictor variables, as it can reduce dimensionality and potentially improve prediction accuracy. However, this comes at the cost of reduced interpretability and increased computational complexity. OLS, on the other hand, is simpler and more interpretable but can struggle with multicollinearity and high-dimensional data.

---

## PCR and PLS


| **Aspect**                    | **Principal Component Regression (PCR)**                                                                                              | **Partial Least Squares Regression (PLS)**                                                                                             |
|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| **Objective**                 | Focuses on explaining the variance in the predictor variables.                                                                        | Focuses on explaining the variance in both the predictor variables and the response variable.                                          |
| **Handling Multicollinearity**| Excellent. Reduces multicollinearity by using uncorrelated principal components derived from predictors.                             | Excellent. Reduces multicollinearity and focuses on components that are most relevant for predicting the response variable.           |
| **Dimensionality Reduction**  | Good. Efficient in reducing the number of variables by using principal components.                                                    | Good. Reduces dimensionality by extracting a small number of latent factors that are useful for predicting the response.              |
| **Interpretability**          | Lower. The principal components are linear combinations of original variables, which may be less interpretable.                      | Moderate. PLS components are also linear combinations, but they are constructed with an eye towards predicting the response.          |
| **Prediction Accuracy**       | Potentially high, especially in cases of strong multicollinearity, but not directly focused on predicting the response variable.    | Often better than PCR, especially when the relationship between predictors and response is complex.                                    |
| **Variable Selection**        | Does not perform variable selection; all variables are included in the principal components.                                         | Can implicitly perform variable selection by emphasizing variables more relevant for predicting the response.                          |
| **Model Complexity**          | Variable. Depends on the number of principal components chosen.                                                                      | Variable. Depends on the number of latent factors chosen.                                                                              |
| **Computational Efficiency**  | Generally efficient, but requires an additional step to calculate principal components.                                              | Less efficient than PCR due to the iterative nature of finding the latent factors.                                                     |
| **Applicability**             | Better suited for data exploration and understanding underlying patterns in the predictors.                                          | More suited for predictive modeling where the goal is to predict a response variable.                                                  |
| **Model Selection**           | Requires choosing the number of principal components.                                                                                 | Requires choosing the number of latent variables, which can be more challenging due to the focus on prediction accuracy.              |

Both PCR and PLS are valuable in dealing with high-dimensional data and multicollinearity, but they serve slightly different purposes. PCR is more focused on dimensionality reduction and is used when the primary goal is to understand the structure in the predictor variables. PLS, however, is more focused on prediction and is used when the primary goal is to predict a response variable, often providing better predictive performance when the predictors and response are complexly related.

---

# Some exam questions

## QDA & LDA Algorithm

Quadratic Discriminant Analysis (QDA) is a statistical method used for classification problems where there are $ K $ classes ($ K > 2 $). Here's a description of its underlying model and the decision rule:

### Underlying Model of QDA

1. **Assumptions**:
   - Each class $ k $ follows a multivariate normal (Gaussian) distribution.
   - The classes have different covariance matrices, i.e., each class $ k $ has its own covariance matrix $ \Sigma_k $.
   - The prior probability of each class, $ P(Y = k) $, can be different.

2. **Probability Density Function**:
   - For a given class $ k $, the probability density function of a feature vector $ x $ is given by the multivariate normal distribution:
     
$$
     f_k(x) = \frac{1}{(2\pi)^{p/2}|\Sigma_k|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu_k)^\top \Sigma_k^{-1} (x - \mu_k)\right)
$$

where $ p $ is the number of features, $ \mu_k $ is the mean vector of class $ k $, and $ \Sigma_k $ is the covariance matrix of class $ k $.

3. **Parameters**:
   - For each class $ k $, QDA estimates the mean vector $ \mu_k $ and the covariance matrix $ \Sigma_k $.
   - The prior probabilities $ P(Y = k) $ can be estimated based on the relative frequencies of each class in the training data or can be set based on domain knowledge.

### Decision Rule

> **(a) Describe the underlying model of quadratic discriminant analysis (QDA) for a general K-class problem (K > 2), and write down its decision rule. [7]**

```{admonition} Answer

Quadratic Discriminant Analysis (QDA) is a statistical method used for classification problems where there are $ K $ classes ($ K > 2 $). Here's a description of its underlying model and the decision rule:

### Underlying Model of QDA

1. **Assumptions**:
   - Each class $ k $ follows a multivariate normal (Gaussian) distribution.
   - The classes have different covariance matrices, i.e., each class $ k $ has its own covariance matrix $ \Sigma_k $.
   - The prior probability of each class, $ P(Y = k) $, can be different.

2. **Probability Density Function**:
   - For a given class $ k $, the probability density function of a feature vector $ x $ is given by the multivariate normal distribution:
     
$$
     f_k(x) = \frac{1}{(2\pi)^{p/2}|\Sigma_k|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu_k)^\top \Sigma_k^{-1} (x - \mu_k)\right)
     $$

where $ p $ is the number of features, $ \mu_k $ is the mean vector of class $ k $, and $ \Sigma_k $ is the covariance matrix of class $ k $.

3. **Parameters**:
   - For each class $ k $, QDA estimates the mean vector $ \mu_k $ and the covariance matrix $ \Sigma_k $.
   - The prior probabilities $ P(Y = k) $ can be estimated based on the relative frequencies of each class in the training data or can be set based on domain knowledge.

### Decision Rule

The decision rule in QDA is to assign a new observation $ x $ to the class that maximizes the posterior probability $ P(Y = k | X = x) $. Using Bayes' theorem, this is equivalent to maximizing the following discriminant function for each class $ k $:


$$
\delta_k(x) = -\frac{1}{2} \ln|\Sigma_k| - \frac{1}{2}(x - \mu_k)^\top \Sigma_k^{-1} (x - \mu_k) + \ln P(Y = k)
$$


An observation $ x $ is classified into the class $ k $ for which $ \delta_k(x) $ is the largest.

### Summary
In summary, QDA models each class with its own covariance matrix and mean vector, assuming a Gaussian distribution for the features within each class. The classification is done by computing a quadratic discriminant function for each class and assigning the observation to the class with the highest value of this function. The "quadratic" in QDA refers to the decision boundary between classes being a quadratic function, due to the class-specific covariance matrices.

```

```{admonition} LDA

For Linear Discriminant Analysis (LDA), the derivation of the discriminant function differs from QDA mainly in its assumption about the covariance matrices of the classes. LDA assumes that all classes have the same covariance matrix, which leads to linear decision boundaries. Here's how the discriminant function for LDA is derived:

### Assumptions of LDA

1. **Common Covariance Matrix**:
   - LDA assumes that all classes share the same covariance matrix, denoted as $ \Sigma $. This is in contrast to QDA, where each class has its own covariance matrix $ \Sigma_k $.

2. **Multivariate Normal Distribution**:
   - Similar to QDA, LDA assumes that the data in each class follows a multivariate normal distribution with class-specific mean vectors $ \mu_k $ but a common covariance matrix $ \Sigma $.

### Derivation of the Discriminant Function

1. **Bayes' Theorem**:
   - As with QDA, we start with Bayes' theorem:
     
$$

     P(Y = k | X = x) = \frac{P(X = x | Y = k) \times P(Y = k)}{P(X = x)}
     $$


2. **Logarithmic Transformation**:
   - The likelihood $ P(X = x | Y = k) $ is modeled as a multivariate normal distribution. After taking the logarithm and simplifying (ignoring the constant terms), the discriminant function becomes:
     
$$

     \ln P(Y = k | X = x) = -\frac{1}{2}(x - \mu_k)^\top \Sigma^{-1} (x - \mu_k) + \ln P(Y = k) + \text{constant}
     $$


3. **Simplification for LDA**:
   - Because the covariance matrix $ \Sigma $ is common across all classes, the quadratic term simplifies, resulting in a linear function. Specifically, the term $ x^\top \Sigma^{-1} x $ is constant for all classes and can be ignored for classification purposes.
   - The discriminant function then simplifies to:
     
$$

     \delta_k(x) = x^\top \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^\top \Sigma^{-1} \mu_k + \ln P(Y = k)
     $$


### Decision Rule for LDA

- An observation $ x $ is classified into the class $ k $ that maximizes the discriminant function $ \delta_k(x) $. Mathematically:
  
$$

  \text{Classify } x \text{ to class } k \text{ if } \delta_k(x) > \delta_j(x) \text{ for all } j \neq k
  $$


### Summary

The key difference between LDA and QDA lies in their assumptions about the covariance matrix. LDA's assumption of a common covariance matrix leads to a simpler, linear discriminant function, which results in linear decision boundaries. This makes LDA more robust to overfitting, especially when the sample size is small, compared to QDA, which can model more complex boundaries but requires more data to estimate class-specific covariance matrices accurately.
```

---

## Discriminant Function (QDA)

The discriminant function in Quadratic Discriminant Analysis (QDA) is derived using Bayes' theorem, which relates the conditional and marginal probabilities of random events. The goal is to assign a new observation $ x $ to the class $ k $ that maximizes the posterior probability $ P(Y = k | X = x) $. Let's break down the steps to obtain the discriminant function:

1. **Bayes' Theorem Application**:
   Bayes' theorem states that:
   
$$
   P(Y = k | X = x) = \frac{P(X = x | Y = k) \times P(Y = k)}{P(X = x)}
   $$

   where:
   - $ P(Y = k | X = x) $ is the posterior probability of class $ k $ given observation $ x $.
   - $ P(X = x | Y = k) $ is the likelihood of observing $ x $ in class $ k $, modeled by the multivariate normal distribution.
   - $ P(Y = k) $ is the prior probability of class $ k $.
   - $ P(X = x) $ is the marginal probability of observing $ x $, which acts as a scaling factor and is the same for each class.

2. **Modeling the Likelihood**:
   The likelihood $ P(X = x | Y = k) $ is modeled as a multivariate normal distribution with a mean vector $ \mu_k $ and covariance matrix $ \Sigma_k $ specific to each class $ k $. The probability density function of a multivariate normal distribution is given by:
   
$$
   f(x | \mu_k, \Sigma_k) = \frac{1}{(2\pi)^{p/2}|\Sigma_k|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu_k)^\top \Sigma_k^{-1} (x - \mu_k)\right)
   $$

where $ p $ is the number of features.

3. **Logarithmic Transformation**:
   To simplify calculations and avoid numerical underflow, we take the logarithm of Bayes' formula (ignoring $ P(X = x) $ since it's constant for all classes). The logarithm of the multivariate normal distribution leads to:
   
$$
   \ln f(x | \mu_k, \Sigma_k) = -\frac{1}{2} \ln|\Sigma_k| - \frac{1}{2}(x - \mu_k)^\top \Sigma_k^{-1} (x - \mu_k) - \frac{p}{2} \ln(2\pi)
   $$

   Since $- \frac{p}{2} \ln(2\pi)$ is constant for all classes, it can be omitted for the purpose of classification.

4. **Deriving the Discriminant Function**:
   Incorporating the prior probability $ P(Y = k) $ and the logarithm of the likelihood, the discriminant function for QDA becomes:
   
$$
   \delta_k(x) = -\frac{1}{2} \ln|\Sigma_k| - \frac{1}{2}(x - \mu_k)^\top \Sigma_k^{-1} (x - \mu_k) + \ln P(Y = k)
   $$


The discriminant function $ \delta_k(x) $ thus obtained is used to classify a new observation $ x $ into the class $ k $ that maximizes this function. The quadratic nature of the decision boundaries in QDA arises from the term $ (x - \mu_k)^\top \Sigma_k^{-1} (x - \mu_k) $ in the discriminant function.

---
## Discriminant Function (LDA)

The simplification in Linear Discriminant Analysis (LDA) arises from the assumption that all classes share the same covariance matrix, denoted as $ \Sigma $. This shared covariance matrix allows for simplification of the discriminant function, leading to linear decision boundaries. Let's break down the simplification process:

### Original Discriminant Function

Starting from Bayes' Theorem, the discriminant function for class $ k $ in LDA, after applying logarithm to the multivariate normal distribution and dropping the constant terms, is given as:


$$
\ln P(Y = k | X = x) = -\frac{1}{2}(x - \mu_k)^\top \Sigma^{-1} (x - \mu_k) + \ln P(Y = k) + \text{constant}
$$


### Breaking Down the Quadratic Term

The quadratic term $(x - \mu_k)^\top \Sigma^{-1} (x - \mu_k)$ can be expanded as:


$$
(x - \mu_k)^\top \Sigma^{-1} (x - \mu_k) = x^\top \Sigma^{-1} x - x^\top \Sigma^{-1} \mu_k - \mu_k^\top \Sigma^{-1} x + \mu_k^\top \Sigma^{-1} \mu_k
$$


Notice that $ x^\top \Sigma^{-1} x $ is a term that does not depend on the class $ k $ and will be the same for every class. In the context of classification, where we are interested in comparing the discriminant function values across different classes, this term does not affect the decision and can thus be omitted.

### Simplification to Linear Function

After dropping the constant $ x^\top \Sigma^{-1} x $ term, the discriminant function simplifies to:


$$
\delta_k(x) = x^\top \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^\top \Sigma^{-1} \mu_k + \ln P(Y = k)
$$


Here's what remains in the simplified discriminant function:
- $ x^\top \Sigma^{-1} \mu_k $: This term is linear in $ x $.
- $ \frac{1}{2} \mu_k^\top \Sigma^{-1} \mu_k $: This is a scalar and does not depend on $ x $. It serves as an offset for each class.
- $ \ln P(Y = k) $: This is the natural logarithm of the prior probability of class $ k $.

### Conclusion

The simplification in LDA to a linear function is due to the removal of the constant quadratic term $ x^\top \Sigma^{-1} x $ in the discriminant function. This leads to linear decision boundaries, as the remaining terms in $ \delta_k(x) $ are linear with respect to $ x $. This is in contrast to Quadratic Discriminant Analysis (QDA), where each class has its own covariance matrix, resulting in a quadratic term in the discriminant function that varies across classes, leading to quadratic decision boundaries.

## K means algorithm

> **Describe the K-means algorithm and explain which optimal solution the algorithm is trying to approximate. [7]**

The K-means algorithm is a popular clustering method used in data analysis and machine learning. Its primary objective is to partition a dataset into $ K $ distinct, non-overlapping clusters. Here's a description of the algorithm and the optimal solution it attempts to approximate:

### Description of the K-means Algorithm

1. **Initialization**:
   - Choose $ K $, the number of clusters.
   - Randomly select $ K $ points from the dataset as the initial centroids of the clusters. These can be actual data points or random points within the data space.

2. **Assignment Step**:
   - Assign each data point to the nearest centroid. The "nearest" is usually defined using the Euclidean distance, although other distance metrics can be used.
   - After this step, each point is assigned to exactly one cluster, based on which centroid it is closest to.

3. **Update Step**:
   - Update the centroid of each cluster to be the mean (average) of all points assigned to that cluster.
   - This mean becomes the new centroid of the cluster.

4. **Iteration**:
   - Repeat the Assignment and Update steps until convergence. Convergence is typically defined as a situation where the centroids no longer move significantly or the assignments no longer change.

5. **Output**:
   - The final output is a set of $ K $ centroids and cluster assignments for each data point.

### Optimal Solution K-means Attempts to Approximate

The K-means algorithm seeks to minimize the within-cluster variance, which is the sum of squared distances between each data point and its corresponding centroid. Mathematically, this objective can be expressed as:

$$
\text{Minimize} \sum_{i=1}^{K} \sum_{x \in S_i} ||x - \mu_i||^2
$$

Here:
- $ K $ is the number of clusters.
- $ S_i $ represents the set of data points in the $ i $-th cluster.
- $ \mu_i $ is the centroid of the $ i $-th cluster.
- $ ||x - \mu_i||^2 $ is the squared Euclidean distance between a data point $ x $ and the centroid $ \mu_i $.

### Characteristics and Limitations


- **Local Optima**: K-means is a heuristic algorithm and can converge to local optima. The final solution depends on the initial choice of centroids.
- **Sensitivity to Initialization**: Different initial centroids can lead to different results. Methods like the K-means++ algorithm are used to optimize the initialization process.
- **Euclidean Distance**: Standard K-means uses Euclidean distance, making it most suitable for spherical clusters of similar sizes. It may not perform well with clusters of different shapes and densities.
- **Specification of $ K $**: The number of clusters $ K $ must be specified in advance. Determining the optimal number of clusters is a separate problem and often requires additional methods like the Elbow method or Silhouette analysis.

In summary, K-means is an iterative algorithm that partitions a dataset into $ K $ clusters by minimizing the within-cluster variance. Its simplicity and efficiency make it widely used, although it has limitations such as sensitivity to initial conditions and a tendency to find local optima.

---

**Briefly describe how the K-medoids algorithm di↵ers from the K-means algorithm. Name one advantage of using the K-medoids algorithm over the K-means algorithm. [7]**

The K-medoids algorithm is similar to the K-means algorithm in that both are used for clustering data into $ K $ groups, but there are key differences in their methodologies and one notable advantage of K-medoids over K-means.

### Differences Between K-medoids and K-means

1. **Centroid vs. Medoid**:
   - **K-means**: The centroid (mean) of the data points in a cluster represents the center of the cluster. This centroid is not necessarily a member of the dataset.
   - **K-medoids**: The medoid is used as the center of each cluster. Unlike the centroid in K-means, the medoid is always one of the data points in the dataset. The medoid is chosen to minimize the sum of dissimilarities between itself and all other points in the cluster.

2. **Sensitivity to Outliers**:
   - **K-means**: More sensitive to outliers since the mean can be significantly influenced by extreme values.
   - **K-medoids**: Less sensitive to outliers, as the medoid (being an actual data point) is typically more centrally located within a cluster.

3. **Distance Measures**:
   - **K-means**: Typically uses Euclidean distance to measure the similarity between data points and centroids.
   - **K-medoids**: Can use a variety of distance metrics, not limited to Euclidean distance. This makes it more versatile, especially for non-numeric data.

4. **Algorithm Complexity**:
   - **K-means**: Generally faster and more computationally efficient due to the simplicity of calculating the mean.
   - **K-medoids**: More computationally intensive, especially for large datasets, because it requires evaluating the cost of swapping medoids and non-medoids.


## Agglomerative algorithm

> You are given a dissimilarity matrix D = (Dij )i,j =1,...,n for observations 1, . . . , n. Clearly describe how the group average agglomerative clustering measures dissimilarity between two clusters of observations G and H, where G and H can be represented by two disjoint subsets of {1, . . . , n}. Then, clearly describe an agglomerative hierarchical clustering algorithm for creating a dendrogram. [7]

To explain the group average agglomerative clustering and the process of creating a dendrogram using an agglomerative hierarchical clustering algorithm, let's break down the concepts and steps:

### Group Average Agglomerative Clustering

In group average agglomerative clustering, the dissimilarity between two clusters $ G $ and $ H $, each a subset of the observations $\{1, \ldots, n\}$, is measured using the average dissimilarity between all pairs of observations, where one member of the pair is from cluster $ G $ and the other is from cluster $ H $. 

Given the dissimilarity matrix $ D = (D_{ij}) $ for observations $ 1, \ldots, n $, the dissimilarity between clusters $ G $ and $ H $ is calculated as:

$$
D(G, H) = \frac{1}{|G| \cdot |H|} \sum_{i \in G} \sum_{j \in H} D_{ij}
$$

where $ |G| $ and $ |H| $ are the number of observations in clusters $ G $ and $ H $, respectively.

### Agglomerative Hierarchical Clustering Algorithm

The agglomerative hierarchical clustering algorithm for creating a dendrogram involves the following steps:

1. **Initialization**: Start by treating each observation as a separate cluster. So, you initially have $ n $ clusters, each containing one observation.

2. **Compute Initial Dissimilarities**: Calculate the dissimilarity between each pair of clusters using the provided dissimilarity matrix $ D $.

3. **Agglomeration Loop**:
   - **Find Closest Clusters**: At each step, identify the two clusters that are closest to each other based on the chosen dissimilarity measure (in this case, the group average).
   - **Merge Clusters**: Combine these two closest clusters into a single new cluster.
   - **Update Dissimilarities**: Recalculate the dissimilarities between the new cluster and all other existing clusters. For any pair of clusters, use the group average formula to determine their dissimilarity.
   - **Repeat**: Continue this process of finding and merging the closest clusters, then updating the dissimilarities.

4. **Stop Condition**: This process is repeated until all observations are merged into a single cluster.

5. **Create Dendrogram**: During the agglomeration process, record the pairs of clusters that are merged and the dissimilarity at which they merge. This information is used to construct a dendrogram, a tree-like diagram that illustrates the arrangement of the clusters formed at each stage of the algorithm. The height at which two clusters are joined in the dendrogram represents the dissimilarity between them.

6. **Interpreting the Dendrogram**: The dendrogram can be used to choose a cut-off dissimilarity level that determines the number of clusters. Cutting the dendrogram at a particular height will give a partition of the data into clusters.

In an exam answer, you would outline these steps, emphasizing the key aspects of the group average method for measuring dissimilarity and the iterative nature of the agglomerative hierarchical clustering process, culminating in the construction of a dendrogram.

### Advantage of K-medoids Over K-means

One significant advantage of the K-medoids algorithm over the K-means algorithm is its robustness to noise and outliers. Since K-medoids chooses actual data points as cluster centers (medoids), it is less influenced by outliers compared to K-means, which calculates the mean (centroid) of the cluster. Outliers can significantly skew the mean, but they have much less impact on the medoid, making K-medoids more suitable for datasets where outliers are present or when the data contains noise. Additionally, the use of different distance metrics in K-medoids makes it more flexible for various types of data, including categorical and non-numeric data.

## T or F
> Suppose X1,...,Xn ∈ Rp are independent and identically distributed normal random vectors with a strictly positive definite covariance matrix Σ. Then the sample covariance matrix S, will have no zero eigenvalues with probability 1 as long as n ≥ p.

True. 

The statement that the sample covariance matrix $ S $ will have no zero eigenvalues with probability 1 as long as $ n \geq p $, given that $ X_1, ..., X_n \in \mathbb{R}^p $ are independent and identically distributed normal random vectors with a strictly positive definite covariance matrix $ \Sigma $, is true. Here's the justification:

1. **Sample Covariance Matrix Definition**: 
   - The sample covariance matrix $ S $ for samples $ X_1, ..., X_n $ is given by 

     $$
     S = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \overline{X})(X_i - \overline{X})^T
     $$

     where $ \overline{X} $ is the sample mean.

2. **Properties of Normal Distribution**:
   - Since $ X_1, ..., X_n $ are i.i.d. normal random vectors with covariance matrix $ \Sigma $ (which is strictly positive definite), the distribution of each $ X_i $ is full rank. This implies that each $ X_i $ spans the entire $ p $-dimensional space with probability 1.

3. **Positive Definite Covariance Matrix**:
   - A strictly positive definite matrix, like $ \Sigma $, has all positive eigenvalues. This characteristic implies that the variabilities along all dimensions are positive, and there is no linear dependency among the dimensions of the data.

4. **Rank of Sample Covariance Matrix**:
   - When $ n \geq p $, there are at least as many observations as dimensions. This generally allows the sample covariance matrix $ S $ to be full rank, which means it has rank $ p $. Therefore, it will have no zero eigenvalues.
   - If $ n < p $ (fewer observations than dimensions), the sample covariance matrix $ S $ cannot be full rank and will have zero eigenvalues.

5. **Probability Consideration**:
   - The condition of having no zero eigenvalues in the sample covariance matrix is equivalent to saying that the matrix is positive definite. In the case of normal distributions, as long as the number of observations is at least as large as the number of dimensions, the sample covariance matrix is almost surely positive definite.

In conclusion, as long as $ n \geq p $, the sample covariance matrix $ S $ constructed from i.i.d. normal random vectors with a strictly positive definite covariance matrix $ \Sigma $ will have no zero eigenvalues with probability 1. This is due to the full-rank nature of the data and the positive definiteness of the true covariance matrix $ \Sigma $.

---
### Tree Size
>  Then increasing the tree size by partitioning the terminal nodes further does not always guarantee a net decrease in the overall node impurity.

The statement is true. In the context of regression trees, the overall node impurity is often measured by the residual sum of squares (RSS). The RSS for a regression tree is calculated as the sum of squared differences between the observed values ($Y_i$) and the mean response values ($\hat{c}_l$) for each node, summed across all nodes ($l$) of the tree.

The mean response value $\hat{c}_l$ for a node is the average of the responses ($Y_i$) for all data points ($X_i$) that fall within that node's region ($R_l$). As you partition the tree further by splitting terminal nodes, you create more regions ($R_l$), each with potentially more homogeneous groups of $Y_i$.

However, while further partitioning can reduce the impurity within individual nodes (since the groups become more homogeneous), it does not always guarantee a net decrease in overall node impurity for several reasons:

1. **Overfitting**: As the tree becomes more complex (with more splits), it may start to fit the noise in the data rather than the underlying pattern. This overfitting can lead to an increase in impurity when the model is applied to new, unseen data.

2. **Decreasing Marginal Returns**: Initially, splits are made at points that greatly reduce impurity. However, as the tree grows, further splits may yield less significant reductions in impurity.

3. **Data Sparsity**: With more splits, some nodes may end up with very few data points, making the estimates of $\hat{c}_l$ less reliable and potentially increasing impurity.

In summary, while increasing the size of a regression tree by adding more splits can reduce impurity in individual nodes, it does not necessarily translate to a decrease in overall node impurity due to overfitting, diminishing returns, and issues with data sparsity in highly partitioned trees.


### FA

> When doing inference based on an orthogonal factor analysis model with p observable variables and k factors, the reason we impose at least k(k − 1)/2 many constraints on the factor loading matrix is to make the model more interpretable.

False.

The imposition of at least $ k(k - 1)/2 $ constraints on the factor loading matrix in orthogonal factor analysis is not primarily for making the model more interpretable, but rather for ensuring identifiability of the model. 

In factor analysis, the model is typically set up with $ p $ observable variables and $ k $ latent factors. The factor loading matrix, which describes the relationship between the observed variables and the latent factors, is not uniquely determined without additional constraints. This non-uniqueness arises because there are infinitely many ways to rotate or transform the factor loading matrix that would result in the same covariance structure among the observed variables.

To resolve this issue, constraints are placed on the factor loading matrix. The most common approach is to impose orthogonality constraints, making the factors uncorrelated (orthogonal to each other). The specific requirement of at least $ k(k - 1)/2 $ constraints corresponds to the number of parameters that can be freely varied in a $ k $-dimensional orthogonal rotation. This ensures that the solution to the factor model is identifiable, meaning it can be uniquely determined given the data.

While these constraints can also aid in interpretability to some extent (for instance, by ensuring factors are uncorrelated, it might be easier to interpret them as representing distinct underlying dimensions), the primary reason for their imposition is to achieve identifiability of the factor model.


---
## PLS Algorithm

**Empirical Formulation of the First Partial Least Squares (PLS) Component:**

Given a finite sample $\{Z_i, X_i\}$ for $i = 1, \ldots, n$ from the population, where $Z \in \mathbb{R}$ is the response variable and $X \in \mathbb{R}^p$ is the covariate vector, the first PLS component at the empirical level involves finding a projection vector that maximizes the sample covariance between $Z$ and the projected covariates $X$.

The empirical projection vector, denoted as $\hat{\phi}$, is obtained by solving the following optimization problem:

$$
\hat{\phi} = \arg\max_{\phi: \|\phi\| = 1} \hat{\text{Cov}}(Z, \phi^T X)
$$

where $\hat{\text{Cov}}(Z, \phi^T X)$ is the sample covariance between $Z$ and $\phi^T X$, calculated as:

$$
\hat{\text{Cov}}(Z, \phi^T X) = \frac{1}{n-1} \sum_{i=1}^n (Z_i - \bar{Z})(\phi^T X_i - \phi^T \bar{X})
$$

Here, $\bar{Z}$ and $\bar{X}$ are the sample means of $Z$ and $X$, respectively.

To find $\hat{\phi}$, we typically:

1. Center the data by subtracting the mean from each variable.
2. Compute the sample covariance matrix between $X$ and $Z$.
3. Normalize $\hat{\phi}$ to ensure $\|\hat{\phi}\| = 1$.

The first empirical PLS component is then the projection $\hat{\phi}^T X$, capturing the direction in the covariate space with the strongest correlation to the response variable in the sample.

---