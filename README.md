# Лабораторная работа 3. Линейная регрессия
### Градиентный спуск для линейной регрессии с одной переменной
Нахождение коэффицентов theta0 и theta1:

<img src = "https://github.com/python-advance/sem6-lr3-Yalkinzsun/blob/master/img/gradient_descent.png" height = "200" />

где `h(x[i]) = theta0 + theta1 * x[i]`

**m** - количество элементов выборки

**a** - cкорость обучения

Функция нахождения целевых параметров (метод градиентного спуска):

```Python
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
```
Сравнение графиков, построенных с помощью самостоятельно подобранных коэффициентов theta0 и theta1 и найденных с помощью метода **polyfit** библиотеки numpy:

<img src = "https://github.com/python-advance/sem6-lr3-Yalkinzsun/blob/master/img/plot.png" height = "600" />

### Линейная регрессия с несколькими переменными


Для выборки из файла `web_traffic.tsv` необходимо определить степень полинома (от 1 до 5), наиболее точно описывающая эти  данные. Для каждой модели со степенями полинома от 1 до 5 найти  среднеквадратическую ошибку и определите для какой модели величина  ошибки уменьшается незначительно. 

Функция для нахождения СКО:

```Python
def sq_error(_x, _y, f_x=None):
    squared_error = []
    for i in range(len(_x)):
        squared_error.append((f_x(_x[i]) - _y[i])**2)
return sum(squared_error)
```
Нохождение целевых параметров для полиномов 1-5 степени:

```Python
th1_1, th0_1 = sp.polyfit(np_x, np_y, 1)
th2_2, th1_2, th0_2 = np.polyfit(np_x, np_y, 2)
th3_3, th2_3, th1_3, th0_3 = np.polyfit(np_x, np_y, 3)
th4_4, th3_4, th2_4, th1_4, th0_4 = np.polyfit(np_x, np_y, 4)
th5_5, th4_5, th3_5, th2_5, th1_5, th0_5 = np.polyfit(np_x, np_y, 5)
```
Определение СКО:

```Python
fun1 = lambda x: th1_1*x + th0_1
fun2 = lambda x: th2_2*x**2 + th1_2*x + th0_2
fun3 = lambda x: th3_3*x**3 + th2_3*x**2 + th1_3*x + th0_3
fun4 = lambda x: th4_4*x**4 + th3_4*x**3 + th2_4*x**2 + th1_4*x + th0_4
fun5 = lambda x: th5_5*x**5 + th4_5*x**4 + th3_5*x**3 + th2_5*x**2 + th1_5*x + th0_5

res1 = sq_error(X, Y, fun1)
res2 = sq_error(X, Y, fun2)
res3 = sq_error(X, Y, fun3)
res4 = sq_error(X, Y, fun4)
res5 = sq_error(X, Y, fun5)

print(f"Ср. кв. ошибка (1) составляет = {res1:.3f}")
print(f"Ср. кв. ошибка (2) составляет = {res2:.3f} на {100 - 100*res2/res1:.2f}% лучше (1)")
print(f"Ср. кв. ошибка (3) составляет = {res3:.3f} на {100 - 100*res3/res1:.2f}% лучше (1)")
print(f"Ср. кв. ошибка (4) составляет = {res4:.3f} на {100 - 100*res4/res1:.2f}% лучше (1)")
print(f"Ср. кв. ошибка (5) составляет = {res5:.3f} на {100 - 100*res5/res1:.2f}% лучше (1)")

```
Результат:
```
Ср. кв. ошибка (1) составляет = 317389767.340
Ср. кв. ошибка (2) составляет = 179983507.878 на 43.29% лучше (1)
Ср. кв. ошибка (3) составляет = 139350144.032 на 56.09% лучше (1)
Ср. кв. ошибка (4) составляет = 126972023.679 на 59.99% лучше (1)
Ср. кв. ошибка (5) составляет = 124464714.566 на 60.78% лучше (1)
```
Для модели со степенью полинома 5 СКО уменьшилось незначительно => более оптимальной будет модель со степенью полинома 4

**Получившийся график:**
<img src = "https://github.com/python-advance/sem6-lr3-Yalkinzsun/blob/master/img/plot2.png" height = "600" />


**Предсказание значения целевого параметра при значениях x = list(range(744, 751)) для линейной регрессии (степень полинома 1) и для степени полинома 5**

Предсказание значений y для x = list(range(744, 751)) и нахождение целевых параметров:

```Python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
np_x = np_x.reshape(-1, 1)
np_y = np_y.reshape(-1, 1)
model.fit(np_x, np_y)

x_predict = np.array(list(range(744, 751)))
new_x_predict = x_predict.reshape(-1, 1)
y_predict = model.predict(new_x_predict)
y_predict = y_predict.flatten()

print("Предсказание значения целевого параметра для x = list(range(744, 751))")
new_th1_1, new_th0_1 = sp.polyfit(x_predict, y_predict, 1)
print(f"Полином 1-ой степени: {new_th0_1:.3f}x + {new_th1_1:.3f}")

new_th5_2, new_th4_2, new_th3_2, new_th2_2, new_th1_2, new_th0_2 = np.polyfit(x_predict, y_predict, 5)
print(f"Полином 5-ой степени: {new_th5_2}x^5 + {new_th4_2}x^4 + {new_th3_2:.6f}x^3 + {new_th2_2:.6f}x^2 + {new_th1_2:.6f}x + {new_th0_2:.6f}")
```
Результат:
```
Полином 1-ой степени: 989.025x + 2.596
Полином 5-ой степени: -1.4855973472127424e-13x^5 + 5.548548750085388e-10x^4 + -0.000001x^3 + 0.000619x^2 + 2.364933x + 1023.573703

```
