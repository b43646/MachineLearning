from numpy import *


def loadDataSet():
    '''
    创建测试数据集
    :return: 数据集，分类向量(1->侮辱性文档，0->正常文档)
    '''
    postingList = [['my', 'dog', 'has', 'flea',
                    'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him',
                    'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute',
                    'I', "love", 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how',
                    'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def createVocabList(dataSet):
    '''
    创建词列表，通过集合返回无重复的列表
    :param dataSet: 数据集
    :return: 词列表
    '''
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWord2Vec(vocabList, inputSet):
    '''
    对比词表(vocabList)与输入的测试集(inputSet)，返回词表大小的向量(如果input中有元素存在于词表中，则返回的词表中对应位置为1)
    :param vocabList: 已经训练好的词表
    :param inputSet: 待测测试集
    :return: 基于测试集的匹配列表
    '''
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: {} is not in my Vocabulary!".format(word))
    return returnVec


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        for word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    '''
    计算条件概率，比如，在一段文本已经被判定为侮辱性语句类别的情况，统计每个词的概率(p(W/Ci))
    :param trainMatrix: 训练的矩阵
    :param trainCategory: 训练的分类列表
    :return: 分类0的条件概率P(W/C0),分类1的条件概率P(W/C1),分类1的概率p(C1)
    '''
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''
    返回分类结果 p(C1/W)=p(w|c1)*p(c1)/p(w)
    :param vec2Classify: 基于测试集的匹配列表
    :param p0Vec: 分类0的条件概率P(W/C0)
    :param p1Vec: 分类1的条件概率P(W/C1)
    :param pClass1: 分类1的概率p(C1)
    :return: 分类结果
    '''
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    # 1. 加载数据，回去数据集合分类表
    listOPosts, listClasses = loadDataSet()

    # 2. 通过数据集获取词汇表
    myVocabList = createVocabList(listOPosts)

    # 3. 从文字到词向量集合转变，也就是postinDoc中有的元素，在词汇表列表中置1
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWord2Vec(myVocabList, postinDoc))

    # 4. 计算条件概率
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))

    # 5. 处理测试数据，即文字到向量集合转变
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWord2Vec(myVocabList, testEntry))

    # 6. 获取分类测试结果
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

    # 同上测试
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWord2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []
    classList = []
    fullText = []

    # 1. 加载数据，回去数据集合分类表
    for i in range(1, 26):
        wordList = textParse(open('email/spam/{}.txt'.format(i), 'r').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/{}.txt'.format(i), 'r').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    # 2. 通过数据集获取词汇表
    vocabList = createVocabList(docList)

    # 3. 准备测试和训练集
    trainingSet = [i for i in range(50)]
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    # 4. 从文本中数据构建词向量
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    # 5. 获取条件概率
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))

    # 6. 使用测试集进行测试
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWord2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("error data:", docList[docIndex])
            print("expected class:", classList[docIndex])
            print("Index:", docIndex)
    print('the error rate is : {}'.format(float(errorCount/len(testSet))))


spamTest()

