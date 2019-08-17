from math import log
import operator


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


def splitDataSet(dataSet, axis, value):
    '''
    按照给定特征划分数据集
    :param dataSet: 数据集
    :param axis: 划分数据集的特征
    :param value: 特征值
    :return: 某个特征的数据集
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    '''
    选择最好的数据集划分方式
    1. 首先以最后一列作为基础，计算熵，其索引为-1
    2. 基本原理： 熵越大，分类越多，越细致
    3. 分别从0-（n-1）列来计算熵，并且比较熵的值，从而选出最大的熵值以及分类索引
    :param dataSet: 数据集
    :return: 最佳分类索引(分类特征)
    '''
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, xlabels):
    '''
    创建树
    :param dataSet: 数据集
    :param xlabels: 标签列表
    :return: 树
    '''
    classList = [example[-1] for example in dataSet]
    labels = xlabels[:]

    # 该函数嵌套执行，所以当余下的分类都是相同的类时，返回类名
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 该函数嵌套执行，所以当余下的数据集合只有一列数据元素时，通过majorityCnt函数计算这一列中数目最多的类名
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 选择最优的分类索引
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]

    # 以最优的标签作为树的根
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]

    # 通过集合去重(集合是无重复的数据集，而列表则可以重复)
    uniqueVals = set(featValues)

    # 嵌套执行
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,
                                            bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testvec):
    '''
    使用决策树的分类函数(遍历整个决策树)
    :param inputTree: 决策树字典
    :param featLabels: 标签
    :param testvec: 测试向量，如[1,0]，'no surfacing?' yes-->'flippers?' no
    :return: 返回对应分类标签
    '''
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testvec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testvec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


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
# print(calcShannonEnt(mydat))
# print(splitDataSet(mydat, 0, 0))
# print(splitDataSet(mydat, 0, 1))
# print(chooseBestFeatureToSplit(mydat))

# print(majorityCnt(['yes', 'yes', 'no', 'yes']))

myTree = createTree(mydat, labels)
print(myTree)
# print(list(myTree.keys())[0])
print(classify(myTree, labels, [1, 0]))
print(classify(myTree, labels, [1, 1]))

# storeTree(myTree, 'tree')
# print(grabTree('tree'))