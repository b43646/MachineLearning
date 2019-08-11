from numpy import *
import operator
from os import listdir


def classify0(inX, dataSet, labels, k):
    '''

    KNN(K-近邻算法)实现：
    - 计算已知类别数据集中的点与当前点之间的距离
    - 按照距离递增排序
    - 选择与当前点距离最小的k个点
    - 确定前k个点所在类别的出现频次
    - 返回前k个点出现频率最高的类别作为当前点的预测分类

    :param inX: 输入向量
    :param dataSet: 样本数据集
    :param labels: 样本对应标签
    :param k: 选择最近邻居的数目
    :return: 所属分类的标签
    '''

    # 1. 获取样本数据集的行数
    dataSetRowSize = dataSet.shape[0]

    # 2. 计算输入向量与样本数据集的距离

    # 2.1 使用inX构建与样本数据集合相同行列的数组，并做差值
    diffMat = tile(inX, (dataSetRowSize, 1)) - dataSet

    # 2.2 [(x-x0)^2 + (y-y0)^2]^0.5
    squareDiffMat = diffMat ** 2

    # 对每一行做求和操作
    sqDistances = squareDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5

    # 2.3 对距离数组排序，并按值升序排列，输出对应的索引
    sortedDistIndices = distances.argsort()
    classCount = {}

    # 3. 选取前k个距离值，并统计每个label出现的次数
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        if voteIlabel in classCount.keys():
            classCount[voteIlabel] = classCount.get(voteIlabel) + 1
        else:
            classCount[voteIlabel] = 1
    # 4. 按照第二个值进行排序，降序排列
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros([numberOfLines, 3])

    classLabelVector = []
    index = 0

    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1

    return returnMat, classLabelVector


def autoNorm(dataSet):

    '''
    归一化
    :param dataSet: 样本数据集
    :return: 归一化数据集，最大值与最小值之间的差值，最小值
    '''

    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals

    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]

    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))

    return normDataSet, ranges, minVals


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']

    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))

    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classfierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
    print("You will probably like person: ", resultList[classfierResult - 1])


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        linStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(linStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []

    # 1. 获取已知的样本数据集
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)

    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)

    # 2. 获取测试数据集
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("result: %d, expected result: %d" % (classifierResult, classNumStr))
        if classNumStr != classifierResult: errorCount += 1

    print("the total error count: %d" % (errorCount))
    print("the error rate: %f" % (errorCount / float(mTest)))


def datingClassTest():
    hoRatio = 0.9
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)

    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("Test Result: %d, Expected Result: %d" %(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("Error Rate: %d" %(errorCount/float(numTestVecs)))


def plotTest():
    returnMat, classLabelVector = file2matrix('datingTestSet2.txt')
    print(classLabelVector)
    print(returnMat)

    normMat, ranges, minVals = autoNorm(returnMat)

    print(normMat)
    print(ranges)
    print(minVals)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.scatter(a[:, 1], a[:, 2])
    ax.scatter(returnMat[:, 1], returnMat[:, 2], 15.0 * array(classLabelVector), 15.0 * array(classLabelVector))
    plt.show()


def basicTest():
    group, labels = createDataSet()
    print(classify0([0, 0], group, labels, 3))


handwritingClassTest()
