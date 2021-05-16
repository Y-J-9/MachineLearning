from pandas import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes

# CART树非叶子结点
class TreeNode:

    '''
    @:param lchild 左子树结点
    @:param rchild 右子树结点
    @:param name 所用到的判别字段
    @:param value 判别值，规定等于该判别值进入左子树，不等于该判别值进入右子树
    '''
    def __init__(self, lchild, rchild, name, value):
        self.lchild = lchild
        self.rchild = rchild
        self.name = name
        self.value = value

# CART树叶子结点
class LeafNode:

    '''
    @:param val 该叶子结点所指向的类为val
    '''
    def __init__(self, val):
        self.val = val

# 分类结果存储在二维表中(DataFrame)
class CART:
    """
    @:param col 用于分类的字段
    @:param frame 总二维表
    @:param train 训练集
    @:param test 测试集
    """

    def __init__(self, frame, train, test, col):
        self.frame = frame
        self.train = train
        self.test = test
        self.col = col


    '''
-----------------------------------------------------------------------------------------------------------------------------------------
    绘制根据训练集训练出的CART树的算法
    '''
    def showCART(self):
        window = plt.subplot()
        plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        plt.axis("off")
        root = self.cartTree(self.frame)
        self.travelCart(root, window, 0.5, 0.9, 0.4)
        plt.show()
    '''
    绘制预剪枝CART树
    '''
    def showFrontCART(self):
        window = plt.subplot()
        plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        plt.axis("off")
        root = self.cartFrontTree(self.train, self.test)
        self.travelCart(root, window, 0.5, 0.9, 0.4)
        plt.show()

    '''
    根据CART树预测物品类别
    @:param obj 物品
    '''
    def check(self, obj, root):
        node = root
        while str(type(node)) != "<class 'MachineLearning.DT_Model.CART.LeafNode'>":
            if obj[node.name] == node.value:
                node = node.lchild
            else:
                node = node.rchild
        return node.val

    '''
    遍历CART树绘图的算法
    @:param root CART树根
    @:param x 该结点X坐标
    @:param y 该结点y坐标
    @:param width 两点宽度
    '''
    def travelCart(self, root, window, x, y, width):
        if str(type(root)) == "<class 'MachineLearning.DT_Model.CART.LeafNode'>":
            circle = mpathes.Circle([x, y], 0.03)
            circle.set_color("g")
            plt.text(x - 0.01, y - 0.01, root.val,fontdict={"size": 15})
            window.add_patch(circle)
            #print("这是叶子结点，指向的类为："+root.val)
        else:
            #print("这是根结点，划分属性为："+root.name+",划分取值为"+root.value)
            circle0 = mpathes.Circle([x, y], 0.03)
            circle0.set_color("r")
            plt.text(x - 0.01, y - 0.01, root.name+"\n"+root.value,fontdict={"size": 15})
            window.add_patch(circle0)
            plt.plot([x, x - width/2], [y, y - 0.2])
            self.travelCart(root.lchild, window,x - width/2, y - 0.2, width/2.05)
            plt.plot([x, x + width/2], [y, y - 0.2])
            self.travelCart(root.rchild, window,x + width/2, y - 0.2, width/2.05)

    '''
    计算table中attribute属性的值分布情况
    @:param table 表
    @:param attribute 属性
    @:return 各个value的频率
    '''
    def calFrequ(self,table, attribute):
        numsTable = {}
        for value in table[attribute]:
            if value in numsTable.keys():
                numsTable[value] += 1
            else:
                numsTable[value] = 1
        for key in numsTable.keys():
            numsTable[key] = numsTable[key]/len(table)
        return numsTable

    '''
    计算当前的最优划分属性及取值
    @:param table 表
    '''
    def bestAttri(self,table):
        # 否则开始计算基尼指数，先对表中的每一个非分类字段
        sumMin = 100
        recordValue = None
        recordName = None
        for each in table:
            if each != self.col:
                # print(each+"字段中：")
                # 统计每个取值的最小基尼指数, numsTable = {value:nums}
                numsTable = self.calFrequ(table, each)
                for value in numsTable:
                    valueGini = numsTable[value] / len(table) * self.Gini(table.loc[table[each] == value]) + (
                                1 - numsTable[value]) * self.Gini(table.loc[table[each] != value])
                    if valueGini < sumMin:
                        sumMin = valueGini
                        recordValue = value
                        recordName = each
        return [recordValue,recordName]
    '''
-------------------------------------------------------------------------------------------------------------------------------------------
    @生成CART树
    @:param table 生成该层结点所用的表格
       好瓜  色泽
    0  是  青绿
    1  否  发黄
    2  是  青绿
    '''
    def cartTree(self, table):
        #如果是叶子结点，那么直接生成并返回
        if self.isLeafNode(table):
            return LeafNode(table.iloc[0][self.col])

        # 否则开始计算最优划分属性
        list = self.bestAttri(table)
        return TreeNode(self.cartTree(table.loc[table[list[1]]==list[0]]),self.cartTree(table.loc[table[list[1]]!=list[0]]),list[1],list[0])

    '''
    判断table现在是否可以直接生成叶子结点了
    '''
    def isLeafNode(self, table):
        #print("table是")
        #print(table)
        val = table.iloc[0][self.col]
        for each in table[self.col]:
            if each != val:
                return False
        return True
    '''
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @生成预剪枝CART树
    @:param table 生成该层结点所用的表格
           好瓜  色泽
        0  是  青绿
        1  否  发黄
        2  是  青绿
    '''
    def cartFrontTree(self, table, test):
        numsvalue = self.calFrequ(test, self.col)
        # 未划分前的划分精度
        maxF = self.finePer(numsvalue)
        # 计算最优划分字段及属性
        list = self.bestAttri(test)
        # 划分后的最优划分精度
        maxB = self.finePer(self.calFrequ(test.loc[test[list[1]] == list[0]], self.col)) + self.finePer(
            self.calFrequ(test.loc[test[list[1]] != list[0]], self.col))
        #print(list[1]+str(maxF)+" "+str(maxB))
        if maxF[0] >= maxB[0]:
            # 禁止划分
            return LeafNode(maxF[1])
        else:
            return TreeNode(self.cartFrontTree(table.loc[table[list[1]] == list[0]],test.loc[test[list[1]] == list[0]]),self.cartFrontTree(table.loc[table[list[1]] != list[0]],test.loc[test[list[1]] == list[0]]),list[1],list[0])

    '''
    判断table是否可以划分，通过划分前最大的精度是否大于划分后所有的精度
    '''
    def canDivide(self, table):
        numsvalue = self.calFrequ(table, self.col)
        # 未划分前的划分精度
        maxF = self.finePer(numsvalue)
        # 计算最优划分字段及属性
        list = self.bestAttri(table)
        # 划分后的最优划分精度
        maxB = self.finePer(self.calFrequ(table.loc[table[list[0]] == list[1]], self.col)) + self.finePer(self.calFrequ(table.loc[table[list[0]] != list[1]], self.col))



    '''
    计算当前频度下的最大精度
    '''
    def finePer(self,numsvalue):
        max = 0
        Name = None
        for key in numsvalue.keys():
            if max < numsvalue[key]:
                max = numsvalue[key]
                Name = key
        return [max, Name]



    '''
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @:param D 二维表所表示的集合
       好瓜  色泽
    0  是  青绿
    1  否  发黄
    2  是  青绿
    '''
    def Gini(self, D):
        table = {}
        for each in D[self.col]:
            if each in table.keys():
                table[each] += 1
            else:
                table[each] = 1
        sum = 0
        for key in table.keys():
            sum += (table[key] / len(D[self.col])) ** 2
        return 1 - sum

    '''
    @:param D 二维表所表示的集合
    @:param A 在特征A的条件下,A为字段
    @:param value 在特征A的条件下，value为取值

    def GiniofA(self, D, A, value):
        tem = D.loc[D[A] == value]
        table = {}
        for each in tem[self.col]:
            if each in table.keys():
                table[each] += 1
            else:
                table[each] = 1
        sum = 0
        print(table)
        for key in table.keys():
            sum += (table[key] / len(D)) * self.Gini(D.loc[D[A] == key])
        return sum
    '''