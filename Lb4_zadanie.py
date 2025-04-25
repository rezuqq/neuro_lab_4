import numpy as np
import pandas as pd
import torch
from torch import nn

# n = 23
# if (n % 2) == 1:
#     print('Решите задачу классификации покупателей '
#           'на классы *купит* - *не купит* (3й столбец) по признакам возраст и доход.')
#
#


df = pd.read_csv('dataset_simple.csv')
X = torch.tensor(df.iloc[:, 0:2].values, dtype=torch.float32)
y = torch.tensor(df.iloc[:, 2].values.reshape(-1, 1), dtype=torch.float32)

#стандартизация данных для среднего возраста
X = (X - X.mean(dim=0)) / X.std(dim=0)



# в структуру нашей сети необходимо внести изменения
# гиперболический тангенс в выходном слое теперь нам не подходит,
# т.к. мы ожидаем 0 или 1 на выходе нейронов, то нам подойдет Сигмоида в качестве
# функции активации выходного слоя
class NNet_multiclass(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(nn.Linear(in_size, hidden_size), # слой линейных сумматоров
                                    nn.Tanh(),                       # функция активации
                                    nn.Linear(hidden_size, out_size),
                                    nn.Sigmoid(),
                                    # nn.Softmax(dim=1) # вместо сигмоиды можно использовать softmax
                                    )
    # прямой проход    
    def forward(self,X):
        pred = self.layers(X)
        return pred

# задаем параметры сети
inputSize = X.shape[1]  # количество признаков задачи
hiddenSizes = 10 # число нейронов скрытого слоя
outputSize = 1  # число нейронов выходного слоя равно числу классов задачи

net = NNet_multiclass(inputSize, hiddenSizes, outputSize)

lossFn = nn.MSELoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.05)

epochs = 1000
for i in range(epochs):
    optimizer.zero_grad()  # обнуляем градиенты
    pred = net(X)  # прямой проход - делаем предсказания
    loss = lossFn(pred, y)  # считаем ошибку
    loss.backward()
    optimizer.step()
    if (i + 1) % 100 == 0:
        print(f'Ошибка на {i + 1} итерации: {loss.item()}')

with torch.no_grad():
    pred = net.forward(X)

pred = torch.Tensor(np.where(pred >= 0, 1, -1).reshape(-1, 1))
err = sum(abs(y - pred)) / 2
print('\nОшибка (количество несовпавших ответов): ')
print(err)