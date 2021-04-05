# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import torch

import matplotlib.pyplot as plt

x_data = [1., 2., 3.]
y_data = [2., 4., 6.]

w1 = torch.randn(1, requires_grad=True)
w2 = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
learning_rate = 1e-2


def forward(x):
    return x * x * w1 + x * w2 + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


for epoch in range(1000):
    for x, y in zip(x_data, y_data):
        l_relu = loss(x, y)
        l_relu.backward()
        print('grad:', x, y, w1.grad.item(), w2.grad.item(), b.grad.item())
        w1.data = w1.data - learning_rate * w1.grad.data
        w2.data = w2.data - learning_rate * w2.grad.data
        b.data = b.data - learning_rate * b.grad.data

        w1.grad.data.zero_()
    print("progress:", epoch, l_relu.item())
    print("w1=", w1.item())
    print("w2=", w2.item())
    print("b=", b.item())
    print()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
