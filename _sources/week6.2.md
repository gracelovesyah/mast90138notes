# week6 lec2

## Partial least squares cont.


In Partial Least Squares (PLS), the first component $ T_1 $ is constructed by the equation:

$$ T_k = \Phi_k X $$

Where $ \Phi_k = 1$ and $|Cov(Z, T_1)|$ is as large as possible.


$$
\hat{\phi}_1 = argmax_{\|\phi_1\|=1} \text{cov}(Z, \phi_1^T X)
$$

```{tip}
The reason why we want to maximize the covariance is because we then combine the variables in a way that explains both the dependent variable and the explanatory variables.
```

