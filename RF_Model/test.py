from MachineLearning.RF_Model.random_forest import Forest
from pandas import DataFrame

frame = None
with open("随机森林\\watermelon3_0a_Ch.txt", 'r', encoding="utf-8") as f:
    col = f.readline().split("\n")[0].split(" ")
    datas = []
    while True:
        data = f.readline()
        if not data:
            break
        data = data.split("\n")[0].split(" ")
        for i in range(0, len(data)):
            data[i] = float(data[i])
        datas.append(data)
    frame = DataFrame(data=datas, columns=col)
forest = Forest(frame=frame, col="好瓜")
forest.generateTree(frame)
forest.generateForest()
for each in range(0, len(frame)):
    print(str(forest.predict(frame.iloc[each,0:2]))+" "+str(each))
forest.drawPicture()

