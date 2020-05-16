# coding=utf-8
# Author:Wang
# Date:2020-06-16
# Email:zy1906917@buaa.edu.cn

'''
原始形式：
    数据集：Mnist
    训练集数量：60000
    测试集数量：10000
    ------------------------------
    运行结果：
    正确率：0.8172
    
对偶形式：
    数据集：Mnist（取样）
    训练集数量：2000
    测试集数量：10000
    ------------------------------
    运行结果：
    正确率：0.8023
'''

import numpy as np
import time
import sys
sys.path.append('..')
from utilities.utilities import ShowProcess


def loadData(fileName):
    '''
    加载Mnist数据集
    :param fileName:directory of data
    :return: list of data and label
    '''
    print('start to read data from %s' % fileName)
    # 存放数据及标记的list
    dataArr = []
    labelArr = []
    
    # 按行读取数据，每行的第一个数据为label，其余为data
    fr = open(fileName, 'r')
    for line in fr.readlines():
        # 对每一行数据按','进行分割，返回字段列表
        curLine = line.strip().split(',')
 
        # Mnsit有10类（标签0-9），perceptron是二分类任务，因而以5为分界线转成{1，-1}
        label = 1 if int(curLine[0]) >= 5 else -1
        labelArr.append(label) 
            
        # 数据归一化，消除量纲
        dataArr.append([int(num)/255 for num in curLine[1:]])

    return dataArr, labelArr


def perceptron_original(dataArr, labelArr, iter=50):
    '''
    感知器训练过程（原始形式）
    :param dataArr: list of data
    :param labelArr: list of label
    :param iter: 迭代次数，默认50
    :return: 训练好的权重w和偏差b
    '''
    print('start to train...')
    
    # 将数据和标签转换成矩阵形式（在机器学习中因为通常都是向量的运算，转换称矩阵形式方便运算）
    # 转换后的数据中每一个样本都是行向量
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T
    
    # 获取数据矩阵的大小，为m*n
    m, n = np.shape(dataMat)
    
    # 创建初始权重w和偏置b，初始值全为0。
    w = np.zeros((1, n))
    b = 0
    
    # 初始化步长，也就是梯度下降过程中的学习率，控制梯度下降速率
    h = 0.0001

    process_bar = ShowProcess(iter, 'OK')
    
    for k in range(iter):
        # 显示训练进度
        process_bar.show_process()
        
        # 对于每一个样本进行梯度下降
        # 梯度下降与随机梯度下降
        # 梯度下降，是全部样本都算一遍以后，统一进行一次梯度下降
        # 随机梯度下降，即计算一个样本就针对该样本进行一次梯度下降
        # 两者的差异各有千秋，此处使用随机梯度下降。
        for i in range(m):
            xi = dataMat[i]
            yi = labelMat[i]
            # 判断是否是误分类样本
            if -1 * yi * (w * xi.T + b) >= 0:
                #对于误分类样本，进行梯度下降，更新w和b
                w = w + h *  yi * xi
                b = b + h * yi

    return w, b


def perceptron_dual(dataArr, labelArr, iter=50):
    '''
    感知器训练过程（对偶形式）
    :param dataArr: list of data
    :param labelArr: list of label
    :param iter: 迭代次数，默认50
    :return: 训练好的权重w和偏差b
    '''
    print('start to train...')
    
    # Gram矩阵的计算所需内存开销较大，不用GPU的情况下60000个样本用我的小surface跑会内存溢出..
    # 取前2000个样本
    count = 2000
    # 将data和label转换成矩阵形式，同上
    dataMat = np.mat(dataArr[0:count])
    labelMat = np.mat(labelArr[0:count]).T
    
    # 计算Gram矩阵（如原书中所述，方便训练过程中查找）
    Gram = np.dot(dataMat, dataMat.T)
    
    m, n = np.shape(dataMat)
    
    # 初始化权重w，偏置项b，alpha，和步长（学习率）lr
    w = np.zeros((1, len(dataArr[0])))
    b = np.zeros((1,1))
    alpha = np.zeros((count, 1))
    lr = 0.001
    
    process_bar = ShowProcess(iter, 'OK')
    for k in range(iter):
        # 显示训练进度
        process_bar.show_process()
        
        for i in range(m):
            y_i = labelMat[i]
            
            # 判断是否是误分类样本
            if y_i * (np.dot(Gram[i], np.multiply(alpha, labelMat)) + b) <= 0:
                alpha[i] += lr
                b += lr * y_i
                
    # 由alpha得到权重w(原书公式2.14)
    for i in range(m):
        w += alpha[i] * labelMat[i] * dataMat[i]
    
    return w, b
    


def model_test(dataArr, labelArr, w, b):
    '''
    测试准确率
    :param dataArr:测试集
    :param labelArr: 测试集标签
    :param w: 训练获得的权重w
    :param b: 训练获得的偏置b
    :return: 正确率
    '''
    print('start to test...')
    # 将data和label转换成矩阵形式，同上
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T

    # 获取测试数据集矩阵的大小
    m, n = np.shape(dataMat)
    
    # 错误样本数计数
    errorCnt = 0
    
    for i in range(m):
        xi = dataMat[i]
        yi = labelMat[i]
        
        # 判断是否为误分类样本，如果是，错误样本数计数加1
        result = -1 * yi * (w * xi.T + b)
        if result >= 0: errorCnt += 1
    
    return 1 - (errorCnt / m)

if __name__ == '__main__':
    #获取当前时间，用于计算程序运行时间
    start = time.time()

    #获取训练集数据及标签，测试集数据及标签
    trainData, trainLabel = loadData('../Mnist/mnist_train.csv')
    testData, testLabel = loadData('../Mnist/mnist_test.csv')

    w, b = perceptron_original(trainData, trainLabel, iter = 30)
    # w,b = perceptron_dual(trainData, trainLabel, iter = 50)
    
    #进行测试，获得正确率
    accRate = model_test(testData, testLabel, w, b)

    #获取当前时间，作为结束时间
    end = time.time()
    
    #显示正确率和时长
    print('accuracy rate is %.4f, time span is %.2f:' % (accRate, end - start))

