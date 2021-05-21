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
        self.father = None
        self.loc = None  # 左孩子还是右孩子


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

    def __init__(self, frame=None, train=None, test=None, col=None):
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
        plt.title("完全CART树")
        root = self.cartTree(self.train)
        self.travelCart(root, window, 0.5, 0.9, 0.4)
        plt.savefig("未剪枝CART树")
        plt.show()

    '''
    绘制预剪枝CART树
    '''

    def showFrontCART(self):
        window = plt.subplot()
        plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        plt.axis("off")
        plt.title("CART预剪枝树")
        root = self.cartFrontTree(self.train)
        self.travelCart(root, window, 0.5, 0.9, 0.4)
        plt.savefig("CART预剪枝树")
        plt.show()

    '''
    绘制后剪枝CART树
    '''

    def showAfterCART(self):
        window = plt.subplot()
        plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        plt.axis("off")
        plt.title("CART后剪枝树")
        root = self.cartTree(self.train)
        self.afterCutCART(root)
        self.travelCart(root, window, 0.5, 0.9, 0.4)
        plt.savefig("CART后剪枝树")
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
            plt.text(x - 0.01, y - 0.01, root.val, fontdict={"size": 15})
            window.add_patch(circle)
            # print("这是叶子结点，指向的类为："+root.val)
        else:
            # print("这是根结点，划分属性为："+root.name+",划分取值为"+root.value)
            circle0 = mpathes.Circle([x, y], 0.03)
            circle0.set_color("r")
            plt.text(x - 0.01, y - 0.01, root.name + "\n" + root.value, fontdict={"size": 15})
            window.add_patch(circle0)
            plt.plot([x, x - width / 2], [y, y - 0.2])
            self.travelCart(root.lchild, window, x - width / 2, y - 0.2, width / 2.05)
            plt.plot([x, x + width / 2], [y, y - 0.2])
            self.travelCart(root.rchild, window, x + width / 2, y - 0.2, width / 2.05)

    '''
    计算table中attribute属性的值分布情况
    @:param table 表
    @:param attribute 属性
    @:return 各个value的频率
    '''

    def calFrequ(self, table, attribute):
        numsTable = {}
        for value in table[attribute]:
            if value in numsTable.keys():
                numsTable[value] += 1
            else:
                numsTable[value] = 1
        for key in numsTable.keys():
            numsTable[key] = numsTable[key] / len(table)
        return numsTable

    '''
    计算当前的最优划分属性及取值
    @:param table 表
    '''

    def bestAttri(self, table):
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
        return [recordValue, recordName]

    '''
    广度优先遍历算法，获得后剪枝队列
    @:param root 根结点
    '''

    def BFS(self, root):
        list = []
        father = []
        queue = [root]
        fqueue = [None]
        while len(queue) != 0:
            node = queue.pop(0)
            if str(type(node.lchild)) != "<class 'MachineLearning.DT_Model.CART.LeafNode'>":
                queue.append(node.lchild)
                node.lchild.father = node
                node.lchild.loc = "left"
            if str(type(node.rchild)) != "<class 'MachineLearning.DT_Model.CART.LeafNode'>":
                queue.append(node.rchild)
                node.rchild.father = node
                node.rchild.loc = "right"
            list.append(node)
        return list

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
        # 先计算最优划分属性
        list = self.bestAttri(table)
        # 如果是叶子结点，那么直接生成并返回
        if self.isLeafNode(table):
            return LeafNode(table.iloc[0][self.col])

        # 如果根据最优划分属性无法继续划分了
        if self.canNotDevide(table, list):
            for i in range(0, len(table)):
                if table.iloc[i][list[1]] == list[0]:
                    return LeafNode(table.iloc[i][self.col])

        # 否则是非叶子结点
        return TreeNode(self.cartTree(table.loc[table[list[1]] == list[0]]),
                        self.cartTree(table.loc[table[list[1]] != list[0]]), list[1], list[0])

    '''
    判断table现在是否可以直接生成叶子结点了
    @:param table 表
    '''

    def isLeafNode(self, table):
        # print("table是")
        # print(table)
        val = table.iloc[0][self.col]
        for each in table[self.col]:
            if each != val:
                return False
        return True

    '''
    根据最优划分属性无法继续划分
    @:param list 最优划分属性及取值
    '''

    def canNotDevide(self, table, list):
        if len(table.loc[table[list[1]] != list[0]]) == 0:
            return True
        return False

    '''
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @生成预剪枝CART树
    @:param train 训练集
    '''

    def cartFrontTree(self, train):
        if len(train) != 0:
            # 先根据训练集直接标记根结点,计算预测精度
            root0 = LeafNode(self.maxAccCol(train))
            fre0 = self.predictFre(root0)
            # 再根据最优划分属性标记非叶子结点
            list = self.bestAttri(train)
            root1 = TreeNode(LeafNode(self.maxAccCol(train.loc[train[list[1]]==list[0]])),
                             LeafNode(self.maxAccCol(train.loc[train[list[1]]!=list[0]])),
                             list[1], list[0])
            fre1 = self.predictFre(root1)
            if fre0 >= fre1:
                return root0
            else:
                root1.lchild = self.cartFrontTree(train.loc[train[list[1]]==list[0]])
                root1.rchild = self.cartFrontTree(train.loc[train[list[1]]!=list[0]])
                return root1
        else:
            return None
    '''
    在table中出现最多的分类属性
    '''
    def maxAccCol(self, table):
        numsvalue = self.calFrequ(table, self.col)
        max = 0
        nodename = None
        for key in numsvalue:
            if numsvalue[key] > max:
                nodename = key
                max = numsvalue[key]
        return nodename
    '''
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    对生成的决策树进行后剪枝
    @:param root 决策树根
    '''

    def afterCutCART(self, root):
        list = self.BFS(root)
        set = self.getColAtt()
        for i in range(len(list) - 1, -1, -1):
            # 计算每个结点是否能被剪枝
            preFre = self.predictFre(root)
            father = list[i].father
            loc = list[i].loc
            # 暴力算法，通过假设获取精度
            if father is not None:
                if loc == "left":
                    for each in set.keys():
                        father.left = LeafNode(each)
                        if self.predictFre(root) < preFre:
                            father.left = list[i]
                elif loc == "right":
                    for each in set.keys():
                        father.rchild = LeafNode(each)
                        if self.predictFre(root) < preFre:
                            father.right = list[i]

    '''
    计算以root为根节点的决策树的预测精度
    '''
    def predictFre(self, root):
        sum = 0
        for i in range(0, len(self.test)):
            if self.check(self.test.iloc[i], root) == self.test.iloc[i][self.col]:
                sum += 1
        return sum / len(self.test)

    '''
    获取分类属性
    '''

    def getColAtt(self):
        set = {}
        for each in self.frame[self.col]:
            if each not in set.keys():
                set[each] = 0
        return set

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

