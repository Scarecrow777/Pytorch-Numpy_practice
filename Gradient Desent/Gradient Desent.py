# forward pass
# loss
# backward pass
import os
import torch
import matplotlib.pyplot as plot
from torch import Tensor
import numpy

# 随机创建的训练数据
x = ([1., 2., 3.])
y = ([2.735, 4.3, 6.5])

w = 0


def loss(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = x*w
        cost += (y_pred - y)**2
    return cost

def gradi(xs, ys):
    gradi = 0
    for x, y in zip(xs, ys):
        gradi = 2 * x * (x * w -y)
    return gradi

learning_rate = 0.1

for epoch in range(500):
    # Forward pass
    loss_v = loss(x, y)
    gradi_v = gradi(x, y)
    # compute loss
    print(epoch, loss_v, gradi_v)
    # Backward pass

    w -= learning_rate * gradi_v
    # compute the gradient

print()
print('So the w is ',w)
