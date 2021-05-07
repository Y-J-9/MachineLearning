import numpy as np
import matplotlib.pyplot as plt
from MachineLearning.Linear_Model.linear_regression import fullAttri


def logistic_function(matX, matW, matY, i):
    # the function in logistic_regression
    tem = matX @ matW
    for each in range(0, len(tem)):
        tem[each][0] = 1.0 / (1 + np.exp(float(-tem[each][0]))) - float(matY[each][0])
    return tem


def iterate_function(matX, matW, matY, alpha, i):
    tem = logistic_function(matX, matW, matY, i)
    for each in range(0, len(tem)):
        tem[each][0] *= matX.getA()[each][i]
    sum = tem.getA().cumsum()[-1]
    return matW[i][0] - alpha /len(matX) * sum


def iterate_function0(matX, matW, matY, alpha):
    sum = logistic_function(matX, matW, matY, len(matW) - 1).getA().cumsum()[-1]
    print(sum) #验证迭代函数是否收敛
    return matW[-1][0] - alpha /len(matX) * sum


def droptrapeziod(matX, matY):
    length = len(matX[0].getA()[0])
    a = np.full((length, 1), -1.0)
    alpha = 0.001
    resulta = []
    for time in range(0, 200000):
        resulta.clear()
        for each in range(0, len(a) - 1):
            resulta.append(iterate_function(matX, a, matY, alpha, each))
        resulta.append(iterate_function0(matX, a, matY, alpha))
        for each in range(0, len(resulta)):
            a[each][0] = resulta[each]
    return a

def drawFreePoint(matX,matY,line):
    print("结果是"+str(line))
    result = np.transpose(matY).getA()[0]
    y = np.transpose(matX).getA()[0]
    x = np.transpose(matX).getA()[1]
    k = -line[0][0]/line[1][0]
    b = -line[2][0]/line[1][0]
    print(k)
    print(b)
    plt.title("y=" + str(k) + "x" + str(b))
    x0 = np.linspace(0, 100, 200)
    y0 = k * x0 + b
    plt.plot(x0, y0)
    for each in range(0, len(result)):
        if result[each] == 1:
            plt.plot(x[each], y[each], "bo")
        else:
            plt.plot(x[each], y[each], "o")
    plt.show()
result = fullAttri("D:\\学习资料\\大二下机器学习\\练习\\线性模型练习\\逻辑回归\\ex2data1.txt")
drawFreePoint(result[0], result[1], droptrapeziod(result[0], result[1]))
