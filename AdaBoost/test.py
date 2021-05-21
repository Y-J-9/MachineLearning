from pandas import DataFrame
from MachineLearning.AdaBoost.AdaBoosting import AdaBoost
frame = None
with open("AdaBoost\\watermelon3_0a_Ch.txt", 'r', encoding="utf-8") as f:
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
print(frame)
add = AdaBoost(frame,"好瓜")
add.drawPicture()