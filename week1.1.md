
# Week 1 Lecture 1
## Matrix Calculation
- **Linear Independence**: A set of vectors is linearly independent if no vector in the set is a linear combination of the others. In terms of matrices, the columns of a matrix are linearly independent if the only solution to $ Ax = 0 $ is the trivial solution $ x = 0 $.

- **Rank**: The rank of a matrix is the maximum number of linearly independent column vectors in the matrix. It is also equal to the maximum number of linearly independent row vectors. Rank reveals the dimension of the vector space spanned by the columns (or rows) of the matrix.

- **Trace**: The trace of a square matrix is the sum of the elements on the main diagonal. It is invariant under change of basis and equals the sum of the eigenvalues (counted with their multiplicities).

## 2 REVIEW OF MATRIX PROPERTIES
- **Determinant**: The determinant of a square matrix is a scalar value that is a function of the entries of a square matrix. It provides information about the volume scaling factor of the linear transformation described by the matrix and whether the matrix is invertible.

- **Inverse**: The inverse of a matrix $ A $ is another matrix, denoted as $ A^{-1} $, such that $ A \cdot A^{-1} = A^{-1} \cdot A = I $ where $ I $ is the identity matrix. Not all matrices have inverses. A matrix is invertible if and only if its determinant is not zero ($ |A| \neq 0 $).

  - $ |A^{-1}| = \frac{1}{|A|} $ when $ A $ is invertible.

- **Eigenvalues and Eigenvectors**:
    - An eigenvector of a matrix $ A $ is a non-zero vector $ v $ such that when $ A $ is applied to $ v $, the vector $ v $ is only scaled by a scalar factor $ \lambda $, i.e., $ Av = \lambda v $, where $ \lambda $ is the eigenvalue corresponding to $ v $.
    - The characteristic equation $ |A - \lambda I| = 0 $ is used to find the eigenvalues $ \lambda $.
    - Eigenvectors can be normalized to have norm 1, which means their length or magnitude is 1: $ \|v\| = \sqrt{v^Tv} = 1 $.
    - The determinant of $ A $ is equal to the product of its eigenvalues, denoted by $ \lambda_i $: $ |A| = \prod_i \lambda_i $.
    - The trace of $ A $ is equal to the sum of its eigenvalues: $ \text{tr}(A) = \sum_i \lambda_i $.

- **Invertible Matrix**:
    - A square matrix is invertible (also known as non-singular or non-degenerate) if it has an inverse matrix.
    - If $ A $ and $ B $ are invertible matrices, then the inverse of their product is the reverse product of their inverses: $ (A \cdot B)^{-1} = B^{-1} \cdot A^{-1} $.

- **Spectral Decomposition**: If a matrix $ A $ is diagonalizable, it can be factored into $ A = \Gamma \Lambda \Gamma^T $, where $ \Gamma $ is a matrix whose columns are the eigenvectors of $ A $, $ \Lambda $ is a diagonal matrix with the eigenvalues of $ A $ on the diagonal, and $ \Gamma^T $ is the transpose of $ \Gamma $.

- **Singular Value Decomposition (SVD)**:
    - Any $ m \times n $ matrix $ A $ can be decomposed as $ A = U \Sigma V^T $, where:
        - $ U $ is an $ m \times m $ orthogonal matrix whose columns are the eigenvectors of $ AA^T $.
        - $ \Sigma $ is an $ m \times n $ diagonal matrix with non-negative real numbers on the diagonal, known as singular values.
        - $ V $ is an $ n \times n $ orthogonal matrix whose columns are the eigenvectors of $ A^TA $.
    - The matrices $ U $ and $ V $ are orthogonal, meaning $ U^T U = V^T V = I $, where $ I $ is the identity matrix.
    - The diagonal entries $ \sigma_i $ of $ \Sigma $ are the square roots of the non-zero eigenvalues of $ A^TA $ or $ AA^T $,