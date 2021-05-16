import numpy as np


class LUTriangle:

    # A 为参数矩阵，二维数组(ndarray)
    # b 为结果矩阵，一维数组(ndarray)
    def __init__(self, A, b):
        try:
            A[0][0] + b[0]
        except Exception:
            print("参数列表出错了，请检查参数矩阵是否为二维数组，结果矩阵是否为一维数组")
        self.A = A
        self.b = b
        self.L = np.zeros((len(self.A), len(self.A[0])))
        self.U = np.zeros((len(self.A), len(self.A[0])))
        self.getLU()

    # 根据参数矩阵分解获得LU矩阵(Doolittle分解)
    def getLU(self):
        # LU矩阵初始化
        self.L[0][0] = 1
        for i in range(0, len(self.A)):
            self.L[i][0] = self.A[i][0] / self.A[0][0]
            self.U[0][i] = self.A[0][i]
        for r in range(1, len(self.A)):
            # 选取主元

            max = self.A[r][r]
            loc = r
            for k in range(r, len(self.A)):
                if self.A[k][r] > max:
                    max = self.A[k][r]
                    loc = k
            self.A[r], self.A[loc] = self.A[loc], self.A[r]
            self.b[r], self.b[loc] = self.b[loc], self.b[r]
            self.L[[r, loc]] = self.L[[loc, r]]

            #print(self.L)
            #print(self.U)
            self.L[r][r] = 1
            '''
            self.L[r][0] = self.A[r][0] / self.A[0][0]
            self.U[0][r] = self.A[0][r]
            '''
            for i in range(r, len(self.A)):
                self.U[r][i] = self.A[r][i] - self.USum_help(r, i)
                if i != r:
                    self.L[i][r] = (self.A[i][r] - self.LSum_help(i, r)) / self.U[r][r]

        print(self.L)
        print(self.U)

    # 得到方程组的解
    def getAnswer(self):
        y = np.zeros(len(self.A))
        x = np.zeros(len(self.A))
        y[0] = self.b[0]
        for i in range(1, len(self.A)):
            y[i] = self.b[i] - self.YSum_help(i, y)
        x[-1] = y[-1] / self.U[-1][-1]
        for i in range(len(self.A) - 2, -1, -1):
            x[i] = (y[i] - self.XSum_help(i, len(self.A), x)) / self.U[i][i]
        return x

    def XSum_help(self, i, n, x):
        sum = 0
        for k in range(i, n):
            sum += self.U[i][k] * x[k]
        return sum

    def YSum_help(self, i, y):
        sum = 0
        for k in range(0, i):
            sum += self.L[i][k] * y[k]
        return sum

    def USum_help(self, r, i):
        sum = 0
        for k in range(0, r):
            sum += self.L[r][k] * self.U[k][i]
        return sum

    def LSum_help(self, i, r):
        sum = 0
        for k in range(0, r):
            sum += self.L[i][k] * self.U[k][r]
        return sum
