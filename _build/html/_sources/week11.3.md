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