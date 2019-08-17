from math import log


def calcShannonEnt(dataSet):
    '''
    以最后一列作为分类特征，通过计算所有分类的概率，以获得熵.
    以createDataSet()的测试数据为例,以最后一列作为分类特征的熵计算如下:
    熵 = - (3/5 * log(3/5, 2) + 2/5 * log(2/5, 2))
    :param dataSet: 数据集
    :return: 熵
    '''
    numEntries = len(dataSet)
    labelCounts = {}

    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


mydat, labels = createDataSet()
print(mydat)
print(calcShannonEnt(mydat))
