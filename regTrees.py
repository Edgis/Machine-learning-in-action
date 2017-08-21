#coding=utf-8
'''
Created on Aug 19, 2017
Tree-Based Regression Methods
@author: Edgis
'''
from numpy import *
from tkinter import *
import matplotlib.pyplot as plt
import matplotlib



def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    lines = fr.readlines()
    for line in lines:
        curLine = line.strip().split('\t')
        #将 每一行映射成浮点数
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0], :]
    return mat0, mat1

#求总方差
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

def regLeaf(dataSet):
    return mean(dataSet[:,-1])                      #返回叶节点,回归树中的目标变量的均值


def createTree(dataSet, leafType = regLeaf, errType = regErr, ops=(1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)        #将数据集进行切分
    if feat == None:
        return val
    else:
        retTree = {}
        retTree["spIndex"] = feat
        retTree["spVal"] = val
        lSet, rSet = binSplitDataSet(dataSet, feat, val)
        retTree["left"]  = createTree(lSet, leafType, errType, ops)     #递归切分
        retTree["right"] = createTree(rSet, leafType, errType, ops)
        return retTree

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0] ; tolN = ops[1]
    # 如果剩余特征值的数目为1，那么就不再切分而返回
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    else:
        m , n= shape(dataSet)
        S = errType(dataSet)
        bestS = inf
        bestIndex = 0
        bestValue = 0
        for featIndex in range(n-1): #对特征进行遍历
            for splitVal in set((dataSet[:,featIndex].T.A.tolist())[0]):   #对特征值进行遍历 set()-->convert list to dict
                mat0 , mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
                if (shape(mat0)[0] <tolN ) or (shape(mat1)[0] < tolN):
                    continue
                newS = errType(mat0) + errType(mat1)

                if newS < bestS :
                    bestIndex = featIndex
                    bestValue = splitVal
                    bestS = newS
        if (S - bestS) < tolS:  #假如误差不大，则退出
            return None, leafType(dataSet)
        mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
        if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
            return None, leafType(dataSet)
        return bestIndex, bestValue

def isTree(obj):
    return (type(obj).__name__ == 'dict')

def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right'])/ 2.0

def prune(tree, testData):
    if shape(testData)[0] == 0:
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):# left or right 是树结构
        lSet, rSet = binSplitDataSet(testData, tree['spIndex'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spIndex'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'], 2)) +\
            sum(power(rSet[:,-1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right'])/2.0
        errorMerge = sum(power(testData[:, -1] - treeMean , 2))
        if errorMerge < errorNoMerge :
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree

def linearSolve(dataSet):
    m, n = shape(dataSet)
    # X, Y 格式化
    X = mat(ones((m, n))) ; Y = mat(ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0:
        raise NameError('This matrix is singular, cannot do inverse, ','\n'
                        ,'try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y

def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws ,X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat , 2))

def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n+1)))
    X[:, 1:n+1] = inDat
    return float(X * model)

def treeForeCast(tree, inDat, modelEval = regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inDat)
    if inDat[tree['spIndex']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inDat, modelEval)
        else:
            return modelEval(tree['left'], inDat)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inDat, modelEval)
        else:
            return modelEval(tree['right'], inDat)

def createForeCast(tree, testData, modelEval = regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    print("yHat : ",'\n',yHat)
    return yHat









if __name__ == "__main__":
    """
    testMat = mat(eye(4))
    print("testMat:",'\n',testMat)
    mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    print("mat0",'\n',mat0)
    print("mat1",'\n',mat1)
    """
    """
    myDat = loadDataSet('ex0.txt')
    myDat = mat(myDat)
    print("myDat ",'\n',myDat)
    myMat = mat(myDat)
    print("myMat ",'\n',myMat)
    createTree(myMat)
    print("createTree(myMat) ",'\n',createTree(myMat))
    """
    """
    myDat = loadDataSet('ex2.txt')
    myDat = mat(myDat)
    print("myDat ", '\n', myDat)
    myMat = mat(myDat)
    print("myMat ", '\n', myMat)
    createTree(myMat, ops=(10000,4))
    print("createTree(myMat) ", '\n', createTree(myMat, ops=(10,4)))
    """
    """
    myDat2 = loadDataSet('ex2.txt')
    myDat2 = mat(myDat2)
    myTree = createTree(myDat2, ops=(0,1))
    print("myTree " ,myTree)
    myDatTest = loadDataSet('ex2test.txt')
    myMat2Test = mat(myDatTest)
    print("pruned Tree", prune(myTree, myMat2Test))
    """
    """
    myDat2 = loadDataSet('exp2.txt')
    myMat2 = mat(myDat2)
    xcord1 = myMat2[:, 0]
    ycord1 = myMat2[:, -1]
    print(myMat2)
    myTree = createTree(myMat2, modelLeaf, modelErr, (1, 10))
    """

    """ 
    wsL = mat(myTree['left'])
    wsL = wsL.flatten().A.tolist()[0]
    wsR = mat(myTree['right'])
    wsR = wsR.flatten().A.tolist()[0]
    print("wsL",'\n',wsL)
    print("myTree",'\n',myTree)
    intevalNode = myTree['spVal']
    x1Mat = []
    for index1 in range(0, int(intevalNode*100), 1):
        x1Mat.append(index1 )
    print("x1Mat",'\n',x1Mat)
    y1Mat = wsL * mat(x1Mat)
    print("y1Mat",'\n',y1Mat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    plt.show()
    """
    #回归树
    trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    myTree = createTree(trainMat, ops = (1, 20))
    yHat = createForeCast(myTree, testMat[:,0])
    print('回归树： ',corrcoef(yHat, testMat[:,1], rowvar=0)[0,1])
    #模型树
    myTree1 = createTree(trainMat, modelLeaf, modelErr, (1,20))
    yHat = createForeCast(myTree1, testMat[:,0], modelTreeEval)
    print('模型树： ',corrcoef(yHat, testMat[:,1], rowvar=0)[0,1])
    #标准线性回归
    ws, X, Y = linearSolve(trainMat)
    for i in range(shape(testMat)[0]):
        yHat[i] = testMat[i,0] * ws[1,0] +ws[0,0]
    print('标准线性回归树： ',corrcoef(yHat, testMat[:,1], rowvar=0)[0,1])









