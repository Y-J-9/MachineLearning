from MachineLearning.DT_Model.CART import LeafNode
from MachineLearning.DT_Model.CART import TreeNode
from MachineLearning.DT_Model.CART import CART
import random
import matplotlib.pyplot as plt
import numpy as np

class Forest:
    def __init__(self, frame, col):
        self.set = []
        self.frame = frame
        self.col = col
        self.cart = CART(frame=frame, col=col)

    '''
    绘制森林及分类点散点图（分类属性为2的情况下）
    '''
    def drawPicture(self):
        att = []
        for each in self.frame:
            if each != self.col:
                att.append(each)
        bFrame = self.frame.loc[self.frame[self.col] == 1]#正例点
        sFrame = self.frame.loc[self.frame[self.col] == 0]#负例点
        plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        plt.scatter(x=bFrame[att[0]],y=bFrame[att[1]],marker="+",s=100)
        plt.scatter(x=sFrame[att[0]],y=sFrame[att[1]],marker="*",s=100) #散点图
        self.plot_decision_boundary() #分类图
        plt.title("随机森林的分类结果图")
        plt.xlabel(att[0])
        plt.ylabel(att[1])
        plt.savefig("随机森林的分类结果图.png")
        plt.show()

    '''
    绘制决策边界的函数
    '''
    def plot_decision_boundary(self):
        X = self.frame["密度"]
        Y = self.frame["含糖率"]
        # 设定最大最小值，附加一点点边缘填充
        x_min, x_max = min(X) - 0.1, max(X) + 0.1
        y_min, y_max = min(Y) - 0.1, max(Y) + 0.1
        h = 0.005
        # 图像上的点集
        xx, yy = np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
        for i in range(0, len(yy)):
            tem = self.predict(obj={"密度": xx[0], "含糖率": yy[i]})
            startx = 0
            for j in range(1, len(xx)):
                temp = self.predict(obj={"密度": xx[j], "含糖率": yy[i]})
                if temp != tem:
                    plt.plot([xx[startx], xx[j]], [yy[i], yy[i]], c="g" if tem >= 0 else "r", linewidth=1)
                    tem = temp
                    startx = j
                if j == len(xx) - 1:
                    plt.plot([xx[startx], xx[j]], [yy[i], yy[i]], c="g" if tem >= 0  else "r", linewidth=1)

    '''
    工具函数，根据给出的字典{value:nums}找出出现次数(nums)最多的值(value)
    @:param dic 字典
    '''
    def getMode(self, dic):
        xmean = 0
        max = 0
        for each in dic.keys():
            if dic[each] > max:
                xmean = each
                max = dic[each]
        return xmean

    '''
    工具函数，根据给出的列表获取其出现次数字典
    @:param list 列表
    '''
    def getDic(self, lines):
        dic = {}
        for each in lines:
            if each in dic.keys():
                dic[each] += 1
            else:
                dic[each] = 1
        return dic

    '''
    绘制每棵树的预测结果
    '''
    def generateLine(self, root, lines, xName, yName):
        if str(type(root)) != "<class 'MachineLearning.DT_Model.CART.LeafNode'>":
            if root.name == xName:
                lines["x"].append(root.value)
                self.generateLine(root.lchild, lines, xName, yName)
                self.generateLine(root.rchild, lines, xName, yName)
            else:
                lines["y"].append(root.value)
                self.generateLine(root.lchild, lines, xName, yName)
                self.generateLine(root.rchild, lines, xName, yName)
    '''
    生成森林
    '''
    def generateForest(self):
        for i in range(0, 13):
            set = []
            for i0 in range(0, len(self.frame)//2):
                set.append(random.randint(0, len(self.frame) - 1))
            table = self.frame.iloc[set]
            self.set.append(self.generateTree(table))

    '''
    根据训练出来的随机森林做预测
    '''
    def predict(self, obj):
        sum = 0
        for each in self.set:
            it = int(self.preTree(obj, root=each))
            if it == 0:
                sum -= 1
            else:
                sum += 1
        return sum

    '''
    根据单个决策树进行预测
    '''
    def preTree(self, obj, root):
        node = root
        while str(type(node)) != "<class 'MachineLearning.DT_Model.CART.LeafNode'>":
            if obj[node.name] >= node.value:
                node = node.lchild
            else:
                node = node.rchild
        return node.val

    '''
    连续的最优划分属性试探法
    '''
    def bestAttri(self, table):
        sumMin = 100
        recordValue = None
        recordName = None
        if len(table) == 0:
            return [None,None]
        for each in table:
            if each != self.col:
                # 统计每个取值的最小基尼指数, numsTable = {value:nums}
                numsTable = self.cart.calFrequ(table, each)
                for value in numsTable:
                    valueGini = numsTable[value] / len(table) * self.cart.Gini(table.loc[table[each] >= value]) + (
                            1 - numsTable[value]) * self.cart.Gini(table.loc[table[each] < value])
                    if valueGini < sumMin:
                        sumMin = valueGini
                        recordValue = value
                        recordName = each
        return [recordValue, recordName]

    '''
    根据所给表格生成一棵树的算法
    '''
    def generateTree(self, table):
        # 计算最优划分属性
        list = self.bestAttri(table)
        # 如果是叶子结点，那么直接生成并返回
        if self.cart.isLeafNode(table):
            return LeafNode(table.iloc[0][self.col])

        # 如果根据最优划分属性无法继续划分了
        if self.cart.canNotDevide(table, list):
            for i in range(0, len(table)):
                if table.iloc[i][list[1]] == list[0]:
                    return LeafNode(table.iloc[i][self.col])

        left = table.loc[table[list[1]] >= list[0]]
        right = table.loc[table[list[1]] < list[0]]

        # 否则是非叶子结点
        return TreeNode(self.generateTree(left),
                        self.generateTree(right), list[1], list[0])