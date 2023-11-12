# week11 additional notes

## Lab
- Using the training data, compute a random forest classifier using the function randomForest from the package randomForest, where the goal is to classify an iris flower to its iris variety, using the four explanatory variables. 
- You can compute the PLS components of X using the function plsr from the package pls. 

---


1. **Entropy**:

   Entropy is a measure of impurity in a dataset. For a binary classification problem (two classes, often labeled 0 and 1), the formula for entropy $ H(S) $ of a set $ S $ with respect to class labels is:

   $$ H(S) = -p_1 \log_2(p_1) - p_2 \log_2(p_2) $$

   - $ p_1 $ is the proportion of samples in class 1 in set $ S $.
   - $ p_2 $ is the proportion of samples in class 2 in set $ S $.

   The idea is that if a set is pure (contains only one class), the entropy is 0. If the set is equally split between two classes, the entropy is 1.

2. **Gini Index**:


    In this formulation, the Gini Index for a set $S$ with $K$ classes is given as:

    $$Gini(S) = \sum_{k=1}^{K} p_k(1 - p_k)$$

    - $p_k$ is the proportion of samples in class $k$ in set $S$.

    This version of the Gini Index also measures impurity in a dataset, and it is commonly used in practice as an alternative to the squared proportion formulation. It ranges between 0 (when the set is pure) and 0.5 (when the set is equally divided among all classes, representing maximum impurity).
3. **Misclassification Error**:

   Misclassification Error, as the name suggests, measures the error rate in a dataset. For a binary classification problem, the formula for Misclassification Error $ ME(S) $ of a set $ S $ with respect to class labels is:

   $$ ME(S) = 1 - \max(p_1, p_2) $$

   - $ p_1 $ is the proportion of samples in class 1 in set $ S $.
   - $ p_2 $ is the proportion of samples in class 2 in set $ S $.

   This metric gives you the proportion of samples that are misclassified. A pure set has a misclassification error of 0, and a perfectly impure set has a misclassification error of 0.5.

In practice, decision tree algorithms use these impurity measures at each node to determine how to split the dataset in a way that reduces impurity the most (e.g., maximizing the information gain or reduction in impurity). The chosen measure depends on the algorithm and specific use case, but all three are common ways to assess the "goodness" of a split in a decision tree.

## Out of Bag 
- Additional Resource
    - [Wikipedia](https://en.wikipedia.org/wiki/Out-of-bag_error)
    - [towardsdatascience](https://towardsdatascience.com/what-is-out-of-bag-oob-score-in-random-forest-a7fa23d710)


### OOB MSE Example

To illustrate the calculation of the Out-Of-Bag (OOB) Mean Squared Error (MSE) in a Random Forest with a simple example, let's assume we have four data points (labeled A, B, C, D) and four variables (or features). We will walk through a hypothetical scenario with a small Random Forest consisting of three trees.

### Step 1: Building the Trees with Bootstrapped Samples
- **Tree 1**: Uses data points A, B, and C. (D is out-of-bag)
- **Tree 2**: Uses data points B, C, and D. (A is out-of-bag)
- **Tree 3**: Uses data points A, C, and D. (B is out-of-bag)

### Step 2: Making Predictions for OOB Samples
- **Tree 1** makes a prediction for D.
- **Tree 2** makes a prediction for A.
- **Tree 3** makes a prediction for B.

### Step 3: Calculating MSE for Each OOB Prediction
Let's assume we have the actual values for A, B, and D, and the predictions made by the trees are as follows (just hypothetical numbers):

- Actual value for A = 5, Tree 2's prediction for A = 4
- Actual value for B = 7, Tree 3's prediction for B = 9
- Actual value for D = 3, Tree 1's prediction for D = 2

Now, calculate the squared error for each prediction:

- For A: $ (5 - 4)^2 = 1 $
- For B: $ (7 - 9)^2 = 4 $
- For D: $ (3 - 2)^2 = 1 $

### Step 4: Calculating the OOB MSE
Finally, calculate the OOB MSE by averaging these squared errors:

$$ \text{OOB MSE} = \frac{1 + 4 + 1}{3} = \frac{6}{3} = 2 $$

So, the OOB MSE for this Random Forest model is 2. This is a simplified example, but it illustrates the process. In real-world scenarios, random forests have many trees (often hundreds), and the OOB MSE is computed by aggregating predictions from all trees for which a particular data point is out-of-bag.

```{admonition} Question
What if D is out of bag for more than 1 tree?
```

```{admonition} Answer
:class: dropdown
If the data point D is out-of-bag for more than one tree in a Random Forest, the process for calculating the OOB MSE would involve averaging the predictions from all trees for which D is out-of-bag. Let's modify the previous example to illustrate this scenario:

### Modified Example with Multiple OOB Predictions for D

#### Step 1: Building the Trees with Bootstrapped Samples
- **Tree 1**: Uses A, B, C. (D is out-of-bag)
- **Tree 2**: Uses B, C. (A and D are out-of-bag)
- **Tree 3**: Uses A, B, D. (C is out-of-bag)

#### Step 2: Making Predictions for OOB Samples
- **Tree 1** predicts D.
- **Tree 2** predicts A and D.
- **Tree 3** predicts C.

#### Step 3: Calculating OOB Predictions for D
Assume these predictions for D:
- Tree 1's prediction for D = 2
- Tree 2's prediction for D = 3

And the actual value for D = 3.

#### Step 4: Averaging Predictions for D
First, average the predictions for D:
- Average prediction for D = \( \frac{2 + 3}{2} = 2.5 \)

#### Step 5: Calculating Squared Errors for All OOB Predictions
Now, calculate the squared error for each OOB prediction:
- Assume the actual values for A and C and the respective tree predictions.
- Calculate squared errors as done before.

#### Step 6: Computing the OOB MSE
Include the squared error for D in the average:
- For D: \( (3 - 2.5)^2 = 0.25 \)
- Include the squared errors for A and C.
- Average all squared errors.

### Key Points:
1. **Averaging Predictions**: When a data point is OOB for multiple trees, average the predictions from all those trees before calculating the squared error.
2. **Aggregating Errors**: Include this squared error in the overall calculation of the OOB MSE, along with the squared errors for other OOB predictions.

This approach ensures that the OOB MSE is a robust measure of the model's performance, incorporating predictions from all trees where a particular data point is out-of-bag.
```