# SGD-OGR-Hessian-estimator
SGD (stochastic gradient descent) with OGR - online gradient regression Hessian estimator, Jarek Duda, 2022

For SGD optimization like neural network training, there is a general pursue to exploit second order derivatives, requiring some Hessian estimation. OGR stands for online gradient regression - estimating Hessian from linear regression of gradients, using 4 expontial moving averages: of (theta, g, theta^2 and theta*g). 

The main article: https://arxiv.org/pdf/1901.11457 , focused on 1D evolving parabola model: https://arxiv.org/pdf/1907.07063

Overview of methods: https://www.dropbox.com/s/54v8cwqyp7uvddk/SGD.pdf and talk: https://youtu.be/ZSnYtPINcug

https://github.com/JarekDuda/SGD-OGR-Hessian-estimator/blob/main/SGD-ORG%20basic.nb basic implementation from https://arxiv.org/pdf/1901.11457 with Beale function optimization, leading to the below diagram.

![alt text](https://github.com/JarekDuda/SGD-OGR-Hessian-estimator/blob/main/OGR%20beale.png)
