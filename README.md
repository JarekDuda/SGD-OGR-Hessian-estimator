# SGD-OGR-Hessian-estimator
SGD (stochastic gradient descent) with OGR - online gradient regression Hessian estimator, Jarek Duda, 2022

For SGD optimization like neural network training, there is a general pursue to exploit second order derivatives, requiring some Hessian estimation. OGR stands for online gradient regression - estimating Hessian from MSE linear regression of gradients, using 4 expontial moving averages: of (theta, g, theta^2 and theta*g). 

The main article: https://arxiv.org/pdf/1901.11457 , focused on 1D evolving parabola model: https://arxiv.org/pdf/1907.07063

Talk focused on OGR: https://youtu.be/Ad1YWjQBMBY and its slides: https://www.dropbox.com/s/thvbwyp5mtwcrw4/OGR.pdf

Overview of methods: https://www.dropbox.com/s/54v8cwqyp7uvddk/SGD.pdf and talk: https://youtu.be/ZSnYtPINcug

Mathematica interactive demonstration: https://github.com/JarekDuda/SGD-OGR-Hessian-estimator/blob/main/SGD-ADAM-OGR%20demo.nb

https://github.com/JarekDuda/SGD-OGR-Hessian-estimator/blob/main/SGD-ORG%20basic.nb basic implementation from https://arxiv.org/pdf/1901.11457 with Beale function optimization, leading to the below optimization trajectories, comparison with momentum and ADAM - 2D full 2nd order method:

![alt text](https://github.com/JarekDuda/SGD-OGR-Hessian-estimator/blob/main/OGR%20beale.png)

2nd order for subspace in high dimension - for neural network training d~10 subpsace in dimension D in millions:

![alt text](https://github.com/JarekDuda/SGD-OGR-Hessian-estimator/blob/main/dsOGRtc.png)

Later corr=1 approximation "c" variants: https://github.com/JarekDuda/SGD-OGR-Hessian-estimator/blob/main/cOGR%20variants.nb

![alt text](https://github.com/JarekDuda/SGD-OGR-Hessian-estimator/blob/main/cOGR%20variants.png)
