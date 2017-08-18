#coding=utf-8
from numpy import *

def loadSimpleData():
    datMat = matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    classLabels = [1.0, 1.0 ,-1.0, -1.0, 1.0]
    return datMat, classLabels

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    m = shape(dataMatrix)[0]
    retArray = ones((m, 1))
    #分类正确
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

#单层决策树生成函数
def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    #print(dataMatrix)
    labelMat = mat(classLabels).T
    #print(labelMat)
    m, n = shape(dataMatrix)
    numSteps = 10.0 ; bestStump = {} ; bestClassEst = mat(zeros((m, 1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        #print(dataMatrix)
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = rangeMin + float(j) * stepSize
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr #计算加权错误率
                #print("split: dim %d ,thresh % .2f, thresh ineqal: %s ,the "
                      #"weighted error is %.3f " %(i, threshVal, inequal, weightedError))
                if weightedError < minError :
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst

#基于单层决策树的AdaBoost训练过程
def adaBoostTrainDS(dataArr, classLabels, numlt):
    weakClassArr =[]
    m = shape(dataArr)[0]
    D = mat(ones((m, 1))/m)
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numlt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D.T", D.T)
        alpha = float(0.5 * log((1.0 - error) /max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        #print("weakClassArr:",weakClassArr)
        #print("classEst:", classEst.T)
        #迭代计算 *D*
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D / D.sum()
        #错误率累加计算
        aggClassEst += alpha * classEst
        #print("aggClassEst:",aggClassEst.T)
        #print("sign(aggClassEst):",sign(aggClassEst))
        #print("mat(classLabels).T:",mat(classLabels).T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        print("aggError",aggErrors)
        errorRate = aggErrors.sum()/m
        print("total error:",errorRate,"\n")
        if errorRate == 0.0:
            break
    #print("----weakClassArr", weakClassArr)
    return weakClassArr, aggClassEst

#AdaBoost 分类函数
def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)): # 行数
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print("aggClassEst:",aggClassEst)
    #print("--->",sign(aggClassEst))
    return sign(aggClassEst)

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

def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)
    ySum =0.0 #AUC的值
    numPosClas = sum(array(classLabels) == 1.0)
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLabels) - numPosClas)
    #获取排好序的索引
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]: #tolist() convert Array to List
        if classLabels[index] ==1.0 :
            delX = 0 ; delY = yStep
        else:
            delX = xStep ; delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate');
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("the Area Under the Curve is:",ySum * xStep)





if __name__ == "__main__":
    """
    datMat, classLabels = loadSimpleData()
    D = mat(ones((5, 1)) / 5)
    buildStump(datMat, classLabels, D)
    """
    """
    datArr, labelArr = loadSimpleData()
    classifierArr = adaBoostTrainDS(datArr, labelArr, 30)
    print(classifierArr)
    """
    """
    datArr, labelArr = loadSimpleData()
    classifierArr = adaBoostTrainDS(datArr, labelArr, 30)
    print("classifierArr",classifierArr)
    print("len(classifierArr):",len(classifierArr))
    adaClassify([[5,5], [0,0]], classifierArr)
    print(adaClassify([[5,5], [0,0]], classifierArr))
    """
    """
    datArr, labelArr = loadDataSet("horseColicTraining2.txt")
    classifierArr = adaBoostTrainDS(datArr, labelArr, 10)
    print("classifierArr", classifierArr)
    testArr, testLabelArr = loadDataSet("horseColicTest2.txt")
    prediction10 = adaClassify(testArr, classifierArr)
    print("prediction10:",prediction10)
    errArr = mat(ones((67, 1)))
    totalErr = errArr[prediction10 != mat(testLabelArr).T].sum()
    totalErrRate = totalErr /67
    print("totalErrRate:",totalErrRate)
    """
    datArr, labelArr = loadDataSet("horseColicTraining2.txt")
    classifierArr, aggClassEst = adaBoostTrainDS(datArr, labelArr, 10)
    plotROC(aggClassEst.T, labelArr)












