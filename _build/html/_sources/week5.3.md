# week5 additional notes

Principal component regression is especially used to overcome the problem with collinearity in linear regression by combining explanatory variables to a smaller set of uncorrelated variables.

```{image} ./images/pcr1.png
:alt: pcr
:class: bg-primary mb-1
:width: 800px
:align: center
```

```{image} ./images/pcr2.png
:alt: pcr
:class: bg-primary mb-1
:width: 800px
:align: center
```
```{image} ./images/pcr3.png
:alt: pcr
:class: bg-primary mb-1
:width: 800px
:align: center
```

## Main idea of PCR 

The linear regression model:

$$
Z_i = m(X_i) + \epsilon_i = \beta^TX_i + \epsilon_i
$$

degenerates to:

$$
Z_i =  \beta^T \Gamma  Y + \epsilon_i = m_PC (Y)\Gamma + \epsilon_i
$$

## Boston housing data Example