from numpy import *
import operator


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


group, labels = createDataSet()

print(classify0([0, 0], group, labels, 3))
