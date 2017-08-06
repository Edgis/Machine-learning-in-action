#coding=utf-8
from numpy import *
#from math import log

def loadDataSet():#数据格式
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]#1 侮辱性文字 ， 0 代表正常言论
    return postingList,classVec

def createVocabList(dataSet):
    #创建空集
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    #创建一个长度为 len(vocabList)， 所含元素全为0的向量
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word %s is not in Vocabulary"%word)
    return returnVec

def trainNBO(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix) #矩阵行数
    numWords = len(trainMatrix[0])#矩阵列数
    #sum(trainCategory)表示label为1 的数量
    pAbusive = sum(trainCategory) / float(numTrainDocs)#label为1的先验概率p(c1)
    p0Num = ones(numWords) #列数
    p1Num = ones(numWords)#列数
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs): #每一行
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
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0- pClass1)
    if p1 > p0:
        return 1
    else:
        return  0

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt'%i, encoding='gbk', errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt'%i, encoding='gbk', errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))#trainingSet = range(50)#
    testSet = []
    #随机构建训练集
    for i in range(10):
        randIndex = int(random.randint(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNBO(array(trainMat), array(trainClasses))
    errorCount = 0
    #对测试集进行分类
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is :', float(errorCount)/len(testSet))
    
if __name__ == "__main__":
    spamTest()
    '''
     listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    print(sum(listClasses))
    print(listClasses)
    print(myVocabList)
    vec1 = setOfWords2Vec(myVocabList, listOPosts[0])
    vec2 = setOfWords2Vec(myVocabList, listOPosts[3])
    print(vec1)
    print(vec2)
    '''
    '''
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V ,pAb = trainNBO(trainMat, listClasses)
    print(p0V)
    print(p1V)
    print(pAb)
    '''
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V ,pAb = trainNBO(trainMat, listClasses)
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry , 'classified as :', classifyNB(thisDoc, p0V, p1V, pAb))



