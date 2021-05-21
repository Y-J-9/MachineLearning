import numpy as np
import math
from MachineLearning.DT_Model.CART import LeafNode
from MachineLearning.DT_Model.CART import TreeNode
import matplotlib.pyplot as plt
class AdaBoost:

    def __init__(self, frame, cal):
        self.frame = frame
        frame["number"] = np.arange(0, len(frame), 1, dtype=int)
        self.weight = np.linspace(1/(len(frame)),1/len(frame),len(frame))
        self.cal = cal

    '''
    经典boosting算法
    '''
    def boosting(self):
        Falpha = [] # 结果的系数列表
        FG = [] # 结果的学习器列表
        while True:
            base = self.generateBase()
            alpha = self.getalpha(base[1])
            for i in range(0, len(self.weight)):
                self.weight[i] = self.weight[i]*(np.e ** (-alpha*self.frame.iloc[i][self.cal]*self.checkOne(base[0], obj=self.frame.iloc[i])))
            Falpha.append(alpha)
            FG.append(base[0])
            if self.checkAll(Falpha, FG) is True:
                break
        return [Falpha, FG]

    '''
    根据误分类率计算学习器系数
    '''
    def getalpha(self, e):
        return 0.5*math.log((1 - e)/e, math.e)


    '''
    根据决策树根作出单个物品的预测
    @:param root 决策树根
    @:param obj 单个物品
    '''
    def checkOne(self, root, obj):
        if obj[root.name] < root.value:
            return root.lchild.val
        else:
            return root.rchild.val

    '''
    根据目前学习出来的集成器对所有数据进行预测，获取是否全部正确分类
    @:param Falpha 参数列表
    @:param FG 学习器列表
    '''
    def checkAll(self, Falpha, FG):
        for j in range(0, len(self.frame)):
            sum = 0
            for i in range(0, len(Falpha)):
                sum += Falpha[i] * self.checkOne(FG[i], obj=self.frame.iloc[j])
            if sum * self.frame.iloc[j][self.cal] < 0:
                return False
        return True

    '''
    根据给定物品预测
    @:param obj 给定物品
    '''
    def checkOther(self, Falpha, FG, obj):
        sum = 0
        for i in range(0, len(Falpha)):
            sum += Falpha[i] * self.checkOne(FG[i], obj=obj)
        if sum > 0:
            return 1
        else:
            return 0

    '''
    生成boosting算法的基本学习器,在这里使用决策树桩(＞或＜）作为基本学习器,基本学习器需要考虑权值分布
    @:returns root 决策树根，e 分类误差率
    '''
    def generateBase(self):
        name = None
        nvalue = None
        e = 10
        cal0 = None
        cal1 = None
        # 计算带有权值的最优划分属性及取值
        for each in self.frame:
            if each != self.cal and each != "number":
                for value in self.frame[each]:
                    reset = self.Em(each, value)
                    res = reset[0]
                    if res < e:
                        e = res
                        name = each
                        nvalue = value
                        cal0 = reset[1]
                        cal1 = reset[2]
        root = TreeNode(lchild=LeafNode(cal0),rchild=LeafNode(cal1),name=name,value=nvalue)
        return [root, e]

    '''
    计算给定字段与取值，最低的分类误差率
    @:param name 字段
    @:param value 取值
    @:return 
    '''
    def Em(self, name, value):
        # 分类正确与否的标识数组
        correct = np.zeros(len(self.frame))
        table0 = self.frame.loc[self.frame[name] < value]
        table1 = self.frame.loc[self.frame[name] >= value]
        cal0 = self.calFre(table0)
        cal1 = self.calFre(table1)
        for i in range(0, len(table0)):
            if table0.iloc[i][self.cal] != cal0:
                correct[int(table0.iloc[i]["number"])] = 1
        for i in range(0, len(table1)):
            if table1.iloc[i][self.cal] != cal1:
                correct[int(table1.iloc[i]["number"])] = 1
        sum = 0
        for i in range(0, len(correct)):
            sum += correct[i]*self.weight[i]
        return [sum, cal0, cal1]


    '''
    计算给定表中，最优的类别标识符
    '''
    def calFre(self, table):
        numsTable = {}
        for value in table[self.cal]:
            if value in numsTable.keys():
                numsTable[value] += 1
            else:
                numsTable[value] = 1
        max = 0
        value = None
        for key in numsTable.keys():
            if numsTable[key] >= max:
                max = numsTable[key]
                value = key
        return value

    '''
    绘图函数
    '''
    def drawPicture(self):
        att = []
        for each in self.frame:
            if each != self.cal:
                att.append(each)
        bFrame = self.frame.loc[self.frame[self.cal] == 1]  # 正例点
        sFrame = self.frame.loc[self.frame[self.cal] == -1]  # 负例点
        plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        plt.scatter(x=bFrame[att[0]], y=bFrame[att[1]], marker="+", s=100)
        plt.scatter(x=sFrame[att[0]], y=sFrame[att[1]], marker="*", s=100)
        res = self.boosting()
        self.plot_decision_boundary(res)
        plt.xlabel("密度")
        plt.ylabel("含糖率")
        plt.title("Adaboosting")
        plt.savefig("Adaboosting.png")
        plt.show()

    '''
    绘制决策边界的函数
    '''
    def plot_decision_boundary(self,res):
        X = self.frame["密度"]
        Y = self.frame["含糖率"]
        # 设定最大最小值，附加一点点边缘填充
        x_min, x_max = min(X) - 0.1, max(X) + 0.1
        y_min, y_max = min(Y) - 0.1, max(Y) + 0.1
        h = 0.005

        # 图像上的点集
        xx, yy = np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
        print(yy)
        for i in range(0, len(yy)):
            tem = self.checkOther(Falpha=res[0], FG=res[1], obj={"密度": xx[0], "含糖率": yy[i]})
            startx = 0
            for j in range(1, len(xx)):
                temp = self.checkOther(Falpha=res[0], FG=res[1], obj={"密度": xx[j], "含糖率": yy[i]})
                if temp != tem:
                    plt.plot([xx[startx],xx[j]],[yy[i],yy[i]],c="g" if tem == 1 else "r",linewidth=1)
                    tem = temp
                    startx = j
                if j == len(xx) - 1:
                    plt.plot([xx[startx],xx[j]],[yy[i],yy[i]],c="g" if tem == 1 else "r",linewidth=1)


