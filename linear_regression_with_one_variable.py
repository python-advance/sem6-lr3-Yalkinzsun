import numpy as np
import matplotlib.pyplot as plt
import csv


X, Y = [], []
csv_reader = csv.reader(open('ex1data1.csv'))
for line in csv_reader:
    X.append(float(line[0]))
    Y.append(float(line[1]))


def gradient_descent(x, y, alpha, n):
    m = len(x)
    theta_0, theta_1 = 0, 0
    for i in range(n):
        sum_1 = 0
        for i in range(m):
            sum_1 += theta_0 + theta_1 * x[i] - y[i]
        temp1 = theta_0 - alpha * (1 / m) * sum_1

        sum_2 = 0
        for i in range(m):
            sum_2 += (theta_0 + theta_1 * x[i] - y[i]) * x[i]
        temp2 = theta_1 - alpha * (1 / m) * sum_2

        theta_0, theta_1 = temp1, temp2

    return theta_0, theta_1

x1 = [1, 25]
y1 = [0, 0]
th0, th1 = gradient_descent(X, Y, 0.001, len(X))
y1[0] = th0 + x1[0] * th1
y1[1] = th0 + x1[1] * th1


plt.plot(x1, y1, 'g', label = u'С коэффицентами, найденными вручную')
plt.scatter(X, Y, label = u'Исходные данные', color='k')

np_x = np.array(X)
np_y = np.array(Y)
new_th1, new_th0 = (np.polyfit(np_x, np_y, 1)).tolist()

new_y1 = [0, 0]
new_y1[0] = new_th0 + x1[0] * new_th1
new_y1[1] = new_th0 + x1[1] * new_th1
plt.plot(x1, new_y1, label = u'С коэффциентами, найденными с помощью метода polyfit')

print(new_th0, new_th1) #-3.8957808783118555 1.1930336441895935


plt.title('Линейная регрессия с одной переменной. Градиентный спуск')
plt.legend()
plt.ylabel('y')
plt.xlabel('x')
plt.show()
plt.savefig('plot.png')
