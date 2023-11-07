# week5 lec 1
## Correlation Graph

Recall that we have obtained correlation formula [](./review2.0.md)
- [correlation](content:references:label-corr)

$$
\rho(X,Y)= \frac{\text{Cov}(X,Y)}{\sqrt{Var(X)Var(Y)}} = \frac{\gamma \lambda}{\sqrt{\sigma\lambda}} 


$$

$$
R^2 = \frac{\gamma^2 \lambda}{\sigma} 
$$

and squared correlation can be interpreted as proportion of variance of Xij explained by Yik.

### R square
The coefficient of determination, denoted \( R^2 \), is a statistical measure that represents the proportion of the variance for a dependent variable (the amount of hours studied) that's explained by an independent variable (the score on a test) or variables in a regression model. It's a common way to evaluate the goodness of fit of a regression model.

The \( R^2 \) value is related to covariance and variance through the following relationship:

$$
R^2 = \frac{\text{Covariance}^2(X,Y)}{\text{Variance}(X) \times \text{Variance}(Y)}
$$

As the sum of R square = 1, the R square of PC1 + R square of PC2 <= 1. Hence we can represent the graph in a circular manner, where x-axis is cor(Xij,Yi1), and y-axis is cor(Xij,Yi2). 

```{tip}
How to read the graph?

1. Angle: smaller angle -> large correlation with PC1 or PC2
2. Length: long -> large correlation
3. Positive / negative: positive correlation or negative correlation

Both angle and length provide important but different pieces of information. The angle provides information about the relationship between variables, while the length indicates the importance of each variable in the principal components. The order of importance depends on what you are looking for:

If you are more interested in the relationships between variables, then angles (and thus correlations) are more important.
If you are interested in which variables are most significant in explaining variance in the dataset, then length is more important.
```

Note: for variables near the center of the biplot, the length and angle of their vectors are not as meaningful as for those near the periphery. The biplot is most informative for variables with vectors that reach out towards the periphery of the plot, indicating they are well explained by the principal components being plotted.





