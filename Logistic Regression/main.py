import os
import torch as th
import numpy as np
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

x_data = th.tensor([[1.0], [2.0], [3.0]])
y_data = th.tensor([[0.], [0.], [1.]])


class LogisticRegressionModel(th.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = th.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = th.sigmoid(self.linear(x))
        return y_pred

model = LogisticRegressionModel()

criterion = th.nn.BCELoss(size_average =False)
optimizer = th.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x = np.linspace(0, 10, 200)
x_t = th.Tensor(x).view((200, 1))
y_t = model(x_t)
y = y_t.data.numpy()

plt.plot(x, y)
plt.plot([0, 10],[0.5, 0.5], c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show()

plt.plot