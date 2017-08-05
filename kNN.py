#coding=utf-8
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
import os

def classify0(inX, dataset, lables, k):
    #shape返回矩阵的[行数， 列数]
    #shape[0]就是行数
    dataSetSize = dataset.shape[0]
    #将 输入的inX转换为和dataSet一样的矩阵形式
    matri_temp = tile(inX, (dataSetSize, 1))
    #求 inX 到各个 train_data的距离
    diffMat = matri_temp - dataset
    sqDiffMat = diffMat**2
    #平方和
    sqDistance = sqDiffMat.sum(axis = 1)#按行累加 ，axis = 1 表示行
    #对平方和开根号
    distance = sqDistance**0.5
    #按照升序排序，返回的是原数组的下标
    sortedDistIndicies = distance.argsort()
    #创建一个空字典
    classCount = {}
    #统计前k个最近的样本所属类别包含的样本个数
    for i in range(k):
        index = sortedDistIndicies[i]
        votelable = lables[index]
        #classCount.get(votelabel, 0 )返回voteIlabel的值，如果不存在，则返回0
        classCount[votelable] = classCount.get(votelable, 0) + 1
    #按照类别计数结果降序
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True )
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    numOfLines = len(arrayOlines) #文件的行数
    returnMat = zeros((numOfLines, 3)) #创建一个0的矩阵
    #returnMat = [[] * 3] * numOfLines
    classLabelVector = []
    index = 0
    for line in arrayOlines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        #returnMat[index,i] = float(listFromLine[i])
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

#归一化特征
def autoNorm(dataset):
    #每列的最小值
    minVals = dataset.min(0)
    #每列的最大值
    maxVals = dataset.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataset))
    m = dataset.shape[0] #行数
    normDataSet = dataset - tile(minVals, (m,1))
    normDataSet = normDataSet / tile(ranges, (m,1))
    return normDataSet, ranges, minVals

def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwlabels = []
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwlabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' %fileNameStr)
    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s'%fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwlabels,3)
        print("the classifier came back with :%d ,the real answer is :%d"\
              %(classifierResult,classNumStr))
        if(classifierResult != classNumStr):
            errorCount += 1
    print("\n the total number of error is %d"%errorCount)
    print("\n the total number rate is :%f"%(errorCount/float(mTest)))






if __name__ == "__main__":
    handwritingClassTest()
    #testVector = img2vector('testDigits/0_13.txt')
    #print(testVector[0,0:31])
    """
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normData , ranges, minVals = autoNorm(datingDataMat)
    print(normData)
    print('/n')
    print(ranges)
    print('/n')
    print(minVals)
    #print(datingDataMat)
    #print('/n')
    #print(datingLabels[1:20])
    """
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,1], datingDataMat[:,2],
               15.0 * array(datingLabels),
               15.0 * array(datingLabels))
    plt.show()
    """





