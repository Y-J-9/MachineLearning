import numpy as np
import matplotlib.pyplot as plt

# We use a list to save plenty of attribute
attriList = []
'''
def initXY(attriList):
    # init the rectangle of data(X,a),rectangle of result(Y,b),we should try to full attriList first.
    a = np.full((len(attriList), len(attriList[0]) + 1), 1.0)
    b = np.zeros((len(attriList), len(attriList[0])))
    for row in range(0, len(attriList)):
        cal = 0
        for key, value in attriList[row].items():
            a[row][cal] = key
            b[row][cal] = value
            cal += 1
    result = []
    mat1 = np.mat(a)
    mat2 = np.mat(b)
    result.append(mat1)
    result.append(mat2)
    return result
'''

def fullAttri(absolute_path):
    XList = []
    YList = []
    # according to the absolute_path to full our attriList,the file should be made like ".csv"
    with open(absolute_path, "r") as f:
        while True:
            lineX = []
            lineY = []
            line = f.readline()
            if not line:
                break
            reList = line[:-1].split(",")
            lineY.append(float(reList[-1]))
            for i in range(0, len(reList) - 1):
                lineX.append(float(reList[i]))
            lineX.append(1.0)
            XList.append(lineX)
            YList.append(lineY)
    mat1 = np.mat(XList)
    mat2 = np.mat(YList)
    resultList = []
    resultList.append(mat1)
    resultList.append(mat2)
    return resultList

def getFinalW(mat1,mat2):
    # learn the final W of data (solve the best equation)
    return np.linalg.inv(np.transpose(mat1)@mat1)@np.transpose(mat1)@mat2

def drawAccordingtoResult(absolutepath,method):
    # solve the best equation
    attriList = fullAttri(absolutepath)
    if method == 0:
        line = getFinalW(attriList[0],attriList[1])
        plt.title("y=" + str(line.getA()[0][0]) + "x" + str(line.getA()[1][0]))
    elif method == 1:
        line = droptrapezoid(attriList[0],attriList[1])
        plt.title("y=" + str(line[0][0]) + "x" + str(line[1][0]))
    plt.plot(np.transpose(attriList[0])[0],
             np.transpose(attriList[1])[0],
             'bo')
    x = np.linspace(0, 25, 50)
    y = float(line[0][0])*x + float(line[1][0])
    plt.plot(x, y)
    plt.show()

def drawLine(wList):
    x = np.linspace(0, 25, 50)
    y = float(wList[0]) * x + float(wList[1])
    plt.plot(x, y)

def droptrapezoid(mat1,mat2):
    a = np.full((len(mat1[0]) + 1, 1),1.0)
    alpha = 0.02
    resulta = []
    # iterate for 1000 times
    for time in range(0,10000):
        resulta.clear()
        for each in range(0,len(a) - 1):
            resulta.append(iterateMissed(mat1,a,mat2,each,alpha))
        resulta.append(iterateMissed0(mat1,a,mat2,alpha))
        if time % 333 == 0:
            drawLine(resulta)
        for each in range(0,len(resulta)):
            a[each][0] = resulta[each]
    return a

def iterateMissed(matX,matW,matY,i,alpha):
    # iterate function
    a = (matX@matW - matY).getA()
    matX = matX.getA()
    for each in range(0,len(a)):
        a[each][i] = a[each][i]*matX[each][i]
    return matW[i][0] - alpha/len(matX)*a.cumsum()[-1]

def iterateMissed0(matX,matW,matY,alpha):
    sum = (matX@matW-matY).getA().cumsum()[-1]
    return matW[-1][0] - alpha/len(matX)*sum

#drawAccordingtoResult("D:\\学习资料\\大二下机器学习\\练习\\线性模型练习\\线性回归\\ex1data1.txt",0)
#drawAccordingtoResult("D:\\学习资料\\大二下机器学习\\练习\\线性模型练习\\线性回归\\ex1data1.txt",1)

