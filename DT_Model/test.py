from MachineLearning.DT_Model.CART import CART
from pandas import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes


#frame = DataFrame([["是","青绿","皮皮虾"],["否","发黄","皮皮虾"],["是","青绿","龙虾"]],index=[0,1,2],columns=["好瓜","色泽","虾类"])
#print(frame["色泽"])
'''
print(frame)
#print(frame.loc[frame["色泽"]!="青绿"])
#print("\n")
cart = CART(frame,"好瓜")
#print(frame.iloc[0]["好瓜"])
#print(frame["好瓜"][0])
node = cart.cartTree(frame)
print(type(node.lchild))
cart.travelCart(node)
a = 10
print(str(type(a)))
print(str(type(a)) == "<class 'int'>")
'''
def getFrame(path, encoding):
    frame = None
    dataList = []
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        col = f.readline()
        col = col.split("\n")[0]
        col = col.split(",")[1:]
        index = []
        while(True):
            opStr = f.readline()
            opStr = opStr.split("\n")[0]
            if not opStr:
                break
            dataList.append(opStr.split(",")[1:])
            index.append(opStr.split(",")[0])
        frame = DataFrame(dataList,index=index,columns=col)
        return frame
frame = getFrame("D:\\Python\\Code\\EditingData\\MachineLearning\\DT_Model\\决策树\\watermelon.txt", "gbk")
train = getFrame("D:\\Python\\Code\\EditingData\\MachineLearning\\DT_Model\\决策树\\Train_set.txt", "utf-8")
test = getFrame("D:\\Python\\Code\\EditingData\\MachineLearning\\DT_Model\\决策树\\Validation_set.txt", "utf-8")
print(train)
cart = CART(frame,train=train,test=test,col=" 好瓜")

#cart.showCART()
#cart.showFrontCART()
#cart.showAfterCART()
#for each in range(0, len(frame)):
    #print(cart.check(frame.iloc[each]))