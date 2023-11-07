# week4 lec 2

## R Code
The R code provided demonstrates the process of principal component analysis (PCA) on the `banknote` dataset available within the `mclust` library. Here's a step-by-step explanation of what the code is doing:

1. **Loading the `mclust` library and dataset:**
    ```r
    library(mclust)
    data(banknote)
    ```
    This loads the necessary `mclust` package and the `banknote` dataset into the R environment.

2. **Plotting the data:**
    ```r
    plot(banknote[,2:7])
    ```
    This command creates a scatterplot matrix for columns 2 to 7 of the `banknote` dataset. Each scatterplot shows the relationship between two variables.

3. **Extracting the `Status` variable:**
    ```r
    StatusX=banknote$Status
    ```
    Here, the `Status` column from the `banknote` dataset is assigned to the variable `StatusX`.

4. **Performing PCA using the `prcomp` function:**
    ```r
    PCX=prcomp(banknote[,2:7])
    ```
    PCA is applied to columns 2 to 7 of the `banknote` dataset using the `prcomp` function, which is a standard function in R for PCA. The default arguments are used, which means the data will be centered but not scaled, and all principal components will be returned.

5. **Extracting the results of PCA:**
    ```r
    PCX
    ```
    This line prints the summary of the PCA object `PCX` to the console, which includes the standard deviations of the principal components, the rotation matrix (loadings), and the scores.

6. **Extracting the rotation matrix and variances:**
    ```r
    G=PCX$rotation
    ell=PCX$sdevˆ2
    ```
    `G` is assigned the rotation matrix, which contains the loadings of the principal components, and `ell` is assigned the variances (squared standard deviations) of the principal components.

   ```r
   Rotation (n x k) = (6 × 6) :
             PC1       PC2       PC3       PC4       PC5       PC6
   Length   0.044    -0.011    0.326   -0.562   -0.753    0.098
   Left    -0.112    -0.071    0.259   -0.455    0.347   -0.767
   Right   -0.139    -0.066    0.345   -0.415    0.535    0.632
   Bottom  -0.768     0.563    0.218    0.186   -0.100   -0.022
   Top     -0.202    -0.659    0.557    0.451   -0.102   -0.035
   Diagonal 0.579     0.489    0.592    0.258    0.084   -0.046
   ```

    The variable `ell` in the context of PCA in R is typically used to store the variances of each principal component. Since it was described as being calculated from the standard deviations (sdev) provided by `PCX$sdev` squared, it will contain the variance explained by each principal component.


    The variances (stored in `ell`) would be the square of `PCX$sdev`


    $$
    \ell = [3.0003, 0.9356, 0.2434, 0.1947, 0.0852, 0.0355]
    $$

7. **Centering the data and plotting the scores manually:**
    ```r
    XC=scale(banknote[,2:7],scale=FALSE) #center data for plot
    pX=XC%*%G
    plot(pX[,1],pX[,2],pch="*",xlab="PC1",ylab="PC 2",asp=1)
    ```
    `XC` centers the data by subtracting the mean of each column. Then `pX` calculates the scores manually by multiplying the centered data by the rotation matrix `G`. These scores are plotted for the first two principal components.

8. **Plotting the PCA results using the `prcomp` object:**
    ```r
    Y=PCX$x
    plot(Y[,1],Y[,2],pch="*",xlab="PC1",ylab="PC 2",asp=1)
    ```
    `Y` extracts the scores directly from the `prcomp` object `PCX`, and then plots the scores of the first two principal components. The `pch="*"` argument specifies the type of point symbol to use in the plot, and `asp=1` sets the aspect ratio to 1, so that one unit on the x-axis is equal to one unit on the y-axis.

Note: There is a typo in the code: `ell=PCX$sdevˆ2` should be `ell=PCX$sdev^2` (using the correct exponentiation operator `^` in R).


## PCX

The output of the `prcomp` function in R, stored in the object `PCX`, contains several components, including:

- `sdev`: The standard deviations of the principal components.
- `rotation`: The matrix of variable loadings (also called the rotation matrix), which represents the correlation between the original variables and the principal components.
- `x`: The matrix of scores of the original data projected onto the principal components.

Let's break down the provided output:

1. **Standard deviations:**
   ```
   Standard deviations (1, .., p=6):
   [1] 1.7321388 0.9672748 0.4933697 0.4412015 0.2919107 0.1884534
   ```
   These are the standard deviations of each of the principal components (PC1 to PC6). They give you an idea of the amount of variance captured by each principal component. A higher standard deviation means that the component accounts for more variance in the data.

2. **Rotation:**
   ```
   Rotation (n x k) = (6 × 6) :
             PC1       PC2       PC3       PC4       PC5       PC6
   Length   0.044    -0.011    0.326   -0.562   -0.753    0.098
   Left    -0.112    -0.071    0.259   -0.455    0.347   -0.767
   Right   -0.139    -0.066    0.345   -0.415    0.535    0.632
   Bottom  -0.768     0.563    0.218    0.186   -0.100   -0.022
   Top     -0.202    -0.659    0.557    0.451   -0.102   -0.035
   Diagonal 0.579     0.489    0.592    0.258    0.084   -0.046
   ```
   This matrix shows how each original variable contributes to each principal component. For example, the first number in the `Length` row (0.044) is the loading of the `Length` variable on the first principal component (PC1). Loadings close to 1 or -1 indicate a strong positive or negative correlation with the principal component, respectively.

3. **Scores (`PCX$x`):**
   `PCX$x` is the matrix of scores, which are the coordinates of the original data projected onto the principal components space. Each row in `PCX$x` corresponds to a banknote, and each column corresponds to a principal component. The scores tell you where each observation lies in the space defined by the principal components.

To interpret `PCX$x` in the context of your dataset:

- The first column of `PCX$x` (i.e.`PCX$x[,1]`) represents the scores on the first principal component (PC1), which captures the most variance. Observations with high scores on PC1 are those that have high values for the variables that have high positive loadings on PC1 and low values for those with high negative loadings.
- The second column of `PCX$x`(i.e.`PCX$x[,2]`)represents the scores on the second principal component (PC2), which captures the second most variance, and so on for the other components.

The principal components provide a means of reducing the dimensionality of the dataset while retaining as much of the variability in the data as possible. The scores can be used for visualization, clustering, or as inputs to other analytical methods.

### Score?

In principal component analysis (PCA), the "scores" (also referred to as "principal component scores" or "factor scores") are the transformed coordinates of the data points in the new feature space defined by the principal components. They are the representation of each observation in the PCA-transformed space.

Here's how the scores are calculated and what they refer to:

1. **Calculation of Scores:**
   The scores are calculated by projecting the original data points onto the principal components. Mathematically, this is done by centering the original variables (subtracting their means) and then multiplying the centered data by the matrix of loadings (the rotation matrix).

   If $ X $ is the original data matrix with variables in columns and observations in rows, and $ G $ is the matrix of loadings (where each column is a principal component), the scores $ Y $ are given by:

   $$
   Y = (X - \bar{X})G
   $$

   Here, $ \bar{X} $ represents the mean of each column of $ X $, and $ Y $ is the matrix of scores. The ith column of $ Y $ contains the scores for the ith principal component.

2. **Interpretation of Scores:**
   The scores give you the coordinates of each observation in the new space spanned by the principal components. An individual score tells you where the corresponding observation lies along a principal component. 

   - A high positive score on a principal component means that the observation is aligned with the direction of the principal component, which in turn means it has high values for the original variables that have high positive loadings and low values for those with high negative loadings on that component.
   - A high negative score indicates the opposite – the observation is aligned in the opposite direction of the principal component.
   - Scores near zero indicate that the observation is close to the mean of the data in the direction of that principal component.

The first principal component is the direction in the data that maximizes variance, so scores on the first component tell you about the primary structure of the data. The second component is orthogonal to the first and represents the next highest variance, and so on for subsequent components.


### PCX and G
`G=PCX$rotation` and `PCX` refer to different components of the PCA analysis in R:

1. **`G=PCX$rotation`:**
   - `G` is the matrix of loadings (also called the rotation matrix) which results from the PCA.
   - The rotation matrix `G` contains the eigenvectors of the covariance matrix of the original data. Each column in `G` corresponds to a principal component.
   - The elements in the matrix represent the correlation of each original variable with each principal component. High absolute values indicate that a variable has a strong correlation with the principal component.

2. **`PCX`:**
   - `PCX` is the entire object that is returned by the `prcomp` function when performing PCA.
   - This object includes several components:
     - `sdev`: The standard deviations of the principal components, indicating the square root of the eigenvalues of the covariance matrix, which represent the amount of variance captured by each component.
     - `rotation`: The same as `G`, this is the matrix of loadings/eigenvectors.
     - `x`: The matrix of principal component scores, which are the original data points projected onto the principal components.
     - `center`: The means of the original variables, which were subtracted from the data during the centering step.
     - `scale`: If scaling was performed, the scaling applied to each variable.
     - `Other elements`: Such as the method used and additional information about the PCA.

In summary, `G` is just one component of the PCA result, focusing on the relationship between the original variables and the principal components, while `PCX` is the comprehensive result object containing all the information about the PCA, including `G` as one of its parts.

## Interpretation of PCs


For the first principal component (PC1):

$$
Yi1 = 0.044 XC_{i1} - 0.112 XC_{i2} - 0.139 XC_{i3} - 0.768 XC_{i4} - 0.202 XC_{i5} + 0.579 XC_{i6}
$$

For the second principal component (PC2):

$$
Yi2 = -0.011 XC_{i1} - 0.071 XC_{i2} - 0.066 XC_{i3} + 0.563 XC_{i4} - 0.659 XC_{i5} + 0.489 XC_{i6}
$$

These equations represent the linear combinations of the centered variables that make up the first two principal components' scores for each observation $ i $.


The interpretation of the principal components (PCs) is based on the magnitude and sign of the loadings (coefficients) of the original variables on each PC. Let's look at the coefficients you've provided for PC1 and PC2:

For PC1:
- The coefficient for the 6th variable (length of diagonal) is 0.579.
- The coefficient for the 4th variable (bottom margin) is -0.768.

For PC2:
- The coefficient for the 5th variable (top margin) is -0.659.
- The coefficient for the 6th variable (length of diagonal) is 0.489.
- The coefficient for the 4th variable (bottom margin) is 0.563.

Here's why the given interpretation makes sense:

1. **First PC (PC1):**
   The largest magnitude coefficients in PC1 are for the 4th and 6th variables, but they have opposite signs. This means that PC1 captures the contrast between these two variables: the length of the diagonal and the bottom margin. When the length of the diagonal is larger (a positive contribution to the score), and the bottom margin is smaller (a negative contribution to the score), the score on PC1 will be higher, and vice versa. Thus, PC1 can be roughly interpreted as corresponding to the difference between the 6th and the 4th variables.

2. **Second PC (PC2):**
   The largest magnitude coefficients in PC2 are also associated with the 4th, 5th, and 6th variables. The 5th variable (top margin) has a negative coefficient, while the 4th and 6th variables have positive coefficients. This suggests that PC2 captures the contrast between the top margin and a combination of the length of the diagonal and the bottom margin. When the top margin is smaller (a negative contribution to the score) and the sum of the length of the diagonal and the bottom margin is larger (positive contributions to the score), the score on PC2 will be higher.

The PCs are linear combinations of the original variables, and their interpretation is often subjective and based on the domain knowledge of the data. When certain variables dominate a principal component due to their large coefficients, it's common to interpret the component as representing those variables or their contrasts, especially when the signs of the loadings are different.

---
## Screeplot
In R, we can apply screeplot to the result of a PC analysis obtained
through prcomp: screeplot(PCX,type="lines")

```{admonition} why using screeplot?
:class: dropdown
A scree plot is used to determine the number of principal components (PCs) to retain in a principal component analysis (PCA) because it provides a visual method to identify the point at which the marginal gain in explained variance decreases significantly. The plot displays the eigenvalues associated with each principal component in descending order, and typically, the eigenvalues will start large and diminish to near zero. The point where the values start to level off—often referred to as the "elbow"—indicates that subsequent principal components contribute less to the explanation of the variance in the data. 

By selecting the number of components before this drop-off point, one can effectively reduce the dimensionality of the data while still capturing the majority of its variability. This is essential in avoiding overfitting and in making the data more interpretable and manageable. The scree plot is a heuristic tool, and its effectiveness lies in its simplicity and the intuitive visual cue it provides for making an informed decision about the number of principal components to retain.
```