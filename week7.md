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

Remember, this table provides a concise view. For a more comprehensive understanding, you'd want to dive deeper into the individual properties and behaviors of each method.