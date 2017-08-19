#coding=utf-8
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    fr = open(fileName)
    numFeat = len(fr.readline().split('\t'))
    dataMat = [] ; labelMat = []
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegres(xArr, yArr):
    xMat = mat(xArr) ; yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0:
        print("This matrix is singular, can not do inverse")
    ws = xTx.I * (xMat.T * yMat)
    return ws

def lwlr(testPoint, xArr, yArr, k = 1.0):
    xMat = mat(xArr) ; yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    #创建权重对角矩阵
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 *k **2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("this matrix is singular, cannot do inverse")
    ws = xTx.I * (xMat.T * (weights * yMat))
    return  testPoint * ws

def lwlrTest(testArr, xArr, yArr, k = 1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return  yHat

def rssError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()

def ridgeRegres(xMat, yMat, lam = 0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0:
        print("This Matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr):
    xMat = mat(xArr) ; yMat = mat(yArr).T
    yMean = mean(yMat, 0)#对列求均值
    #数据标准化
    yMat = yMat -yMean
    xMeans = mean(xMat, 0) #对 列 求均值
    xVar = var(xMat, 0)#对列求方差
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i-10))
        wMat[i,:] = ws.T
    return wMat

def regularize(xMat):
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)
    inVar = var(inMat, 0)
    inMat = (inMat - inMeans)/inVar
    return inMat


def stageWise(xArr, yArr, eps = 0.01, numIt = 100):
    xMat = mat(xArr) ; yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat -yMean
    xMat = regularize(xMat)
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))
    ws = zeros((n, 1)) ; wsTest = ws.copy() ; wsMax = ws.copy()
    for i in range(numIt):
        print("ws.T: ",ws.T)
        lowestError = inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError :
                    lowestError = rssE
                    wsMax = wsTest
        ws = ws.copy()
        returnMat[i, :] = ws.T
    return returnMat


if __name__ == "__main__":
    """
    xArr, yArr = loadDataSet('ex0.txt')
    ws = standRegres(xArr,yArr)
    print(ws)
    xMat = mat(xArr)
    yMat = mat(yArr)
    yHat = xMat * ws
    """
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()
    """
    #print(corrcoef(yHat.T, yMat))
    """
    xArr, yArr = loadDataSet('ex0.txt')
    print("actual yArr[0]:",yArr[0])
    lwlr(xArr[0], xArr, yArr, 1.0)
    print(lwlr(xArr[0], xArr, yArr, 1.0))
    """
    """
    xArr, yArr = loadDataSet('ex0.txt')
    print(lwlrTest(xArr, xArr, yArr, 0.003  ))
    """
    """
    xArr, yArr = loadDataSet('ex0.txt')
    xMat = mat(xArr)
    print("xMat: ",xMat)
    yMat = mat(yArr)
    yHat = lwlrTest(xArr, xArr, yArr, 0.01  )
    srtInd = xMat[:, 1].argsort(0) #返回的是数组值从小到大的索引值, 按列排序
    print("srtInd: ",srtInd)
    xSort = xMat[srtInd][:, 0, :] #从小到大 排序
    print("xSort: ",xSort)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1], yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c = 'red')
    plt.show()
    """
    """
    abX, abY = loadDataSet('abalone.txt')
    ridgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()
    """
    xArr, yArr = loadDataSet('abalone.txt')
    #stageWise(xArr, yArr, 0.001, 5000)
    print(stageWise(xArr, yArr, 0.001,5000))














