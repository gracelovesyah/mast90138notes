# week9 lec 2

## Decision Tree
## Random Forest

## Regression Tree

When to use RT - data's distribution is wired.

```{image} ./images/rt1.png
:alt: rt
:class: bg-primary mb-1
:width: 800px
:align: center
```

How to build RT - use mean value on node, evaluate with RSS
i.e. We decide the splitting criteria based when RSS is minimum.

When to stop splitting (to prevent overfitting)? 
```{admonition} Answer
This can be done by setting the stopping criteria. Which is usually the number of number on each node. Usually 20. So when there are <= 20 samples on a node, we will end splitting and call that a leaf. n 
```


```{image} ./images/rt2.png
:alt: rt
:class: bg-primary mb-1
:width: 800px
:align: center
```


What if multiple features in regression tree?
