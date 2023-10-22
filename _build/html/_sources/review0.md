# Overivew
## MLE
### Additional Resource
- [Maximum Likelihood For the Normal Distribution, step-by-step!!!](https://www.youtube.com/watch?v=Dn6b9fCIUpM)

## Bayesian
In the realm of statistics and machine learning, Bayesian inference is a method of statistical inference where Bayes' theorem is used to update the probability estimate for a hypothesis as more evidence becomes available. This is especially useful in parameter estimation.

Let's take a simple example: Estimating the bias of a coin.

### Setup

Imagine you have a coin, and you want to estimate the probability $ \theta $ of it landing heads. In a frequentist setting, you might flip it many times and compute the fraction of times it lands heads. In a Bayesian setting, you start with a prior belief about $ \theta $ and update this belief with observed data.

### Bayesian Approach

1. **Prior**: Start with a prior distribution over $ \theta $. A common choice for the bias of a coin is the Beta distribution, which is parameterized by two positive shape parameters, $ \alpha $ and $ \beta $. If you have no strong belief about the coin's bias, you might choose $ \alpha = \beta = 1 $, which is equivalent to a uniform distribution over [0, 1].

2. **Likelihood**: This is the probability of observing the data given a particular value of $ \theta $. If you flip the coin $ n $ times and observe $ h $ heads, the likelihood is given by the binomial distribution:

$$ P(data|\theta) = \binom{n}{h} \theta^h (1-\theta)^{n-h} $$

3. **Posterior**: Using Bayes' theorem, the posterior distribution for $ \theta $ after observing the data is:

$$ P(\theta|data) = \frac{P(data|\theta) \times P(\theta)}{P(data)} $$

$$
Posterior == \frac{Likelihood \cdot Prior}{Evidence}
$$

Given the conjugacy between the Beta prior and the Binomial likelihood, the posterior is also a Beta distribution but with updated parameters: $ \alpha' = \alpha + h $ and $ \beta' = \beta + n - h $.

4. **Estimation**: There are different ways to estimate $ \theta $ from the posterior distribution. A common choice is to use the mean of the posterior, which for a Beta distribution is:

$$ \hat{\theta} = \frac{\alpha'}{\alpha' + \beta'} $$

### Intuition

Let's say you start with a completely uniform prior (no knowledge about the coin's bias). After flipping the coin 10 times, you observe 7 heads. Your belief (posterior distribution) about the coin's bias will shift toward 0.7, but it will also consider your initial uncertainty. The more data (coin flips) you observe, the more your belief will be influenced by the observed data relative to the prior.

In Bayesian parameter estimation, rather than obtaining a single "best estimate" value as in frequentist statistics, you obtain a distribution over the possible parameter values that reflects both the data and the prior beliefs.