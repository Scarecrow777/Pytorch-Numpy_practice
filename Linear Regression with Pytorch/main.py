# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

import torch as th

x_data = th.tensor([[1.0], [2.0], [3.0]])
y_data = th.tensor([[2.0], [4.0], [6.0]])


class LinearModel(th.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = th.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()

criterion = th.nn.MSELoss()
optimizer = th.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w= ', model.linear.weight.item())
print('b= ', model.linear.bias.item())

x_test = th.tensor([4.0])
y_test = model(x_test)
print('y_pred =', y_test.data.item())

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
