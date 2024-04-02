# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:09:38 2018

@author: HAN_RUIZHI yb77447@umac.mo OR  501248792@qq.com

This code is the first version of BLS Python. 
If you have any questions about the code or find any bugs
   or errors during use, please feel free to contact me.
If you have any questions about the original paper, 
   please contact the authors of related paper.
"""
import math
import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA 
import time
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import math
'''
#评价指标：accuracy, precision, recall, f1_score, AUC
'''
def show_accuracy(predictLabel,Label): 
    label_1 = np.zeros(Label.shape[0])
    predlabel = []
    label_1 = Label.argmax(axis = 1)
    predlabel = predictLabel.argmax(axis = 1)
    return accuracy_score(label_1, predlabel)


def show_precision(predictLabel,Label): 
    label_1 = np.zeros(Label.shape[0])
    predlabel = []
    label_1 = Label.argmax(axis = 1)
    predlabel = predictLabel.argmax(axis = 1)
    return precision_score(label_1, predlabel, average='weighted')

def show_recall(predictLabel,Label): 
    label_1 = np.zeros(Label.shape[0])
    predlabel = []
    label_1 = Label.argmax(axis = 1)
    predlabel = predictLabel.argmax(axis = 1)
    return recall_score(label_1, predlabel, average='weighted')

def show_f1score(predictLabel,Label): 
    label_1 = np.zeros(Label.shape[0])
    predlabel = []
    label_1 = Label.argmax(axis = 1)
    predlabel = predictLabel.argmax(axis = 1)
    return f1_score(label_1, predlabel, average='weighted')

def show_auc(predictLabel,Label): 
    label_1 = np.zeros(Label.shape[0])
    predlabel = []
    label_1 = Label.argmax(axis = 1)
    predlabel = predictLabel.argmax(axis = 1)
    class_names = np.unique(Label)
    label_1 = label_binarize(label_1, classes=class_names)
    predlabel=label_binarize(predlabel, classes = class_names)
    fpr, tpr, th = roc_curve(label_1.ravel(), predlabel.ravel())
    return auc(fpr, tpr)

'''
激活函数
'''
def kerf( matrix ): # 径向基激活函数
    return np.exp( np.multiply(-1 * matrix, matrix) / 2.0 ) / np.sqrt(2 * math.pi)

def relu6(inputs):
    return tf.minimum(tf.maximum(inputs,0),6)

def h_swish(inputs):
    return tf.multiply(relu6(inputs+3)/6,inputs)

def mish(z): #激活函数
    return z * (np.exp(np.log(1 + np.exp(z))) - np.exp(-np.log(1 + np.exp(z)))) / (np.exp(np.log(1 + np.exp(z))) + np.exp(-np.log(1 + np.exp(z))))

def tansig(x):
    return (2/(1+np.exp(-2*x)))-1

def sigmoid(data):
    return 1.0/(1+np.exp(-data))
    
def linear(data):
    return data
    
def tanh(data):
    return (np.exp(data)-np.exp(-data))/(np.exp(data)+np.exp(-data))
    
def relu(data):
    return np.maximum(data,0)

def pinv(A,reg):
    return np.mat(reg*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)
'''
参数压缩
'''
def shrinkage(a,b):
    z = np.maximum(a - b, 0) - np.maximum( -a - b, 0)
    return z
'''
参数稀疏化
'''
def sparse_bls(A,b):
    lam = 0.001
    itrs = 50
    AA = A.T.dot(A)   
    m = A.shape[1]
    n = b.shape[1]
    x1 = np.zeros([m,n])
    wk = x1
    ok = x1
    uk = x1
    L1 = np.mat(AA + np.eye(m)).I
    L2 = (L1.dot(A.T)).dot(b)
    for i in range(itrs):
        ck = L2 + np.dot(L1,(ok - uk))
        ok = shrinkage(ck + uk, lam)
        uk = uk + ck - ok
        wk = ok   
    return wk

#此时类别是5个类
def transformLabel(label_raw):
    if label_raw == 0: ##
        results = [1, 0, 0, 0, 0]
    elif label_raw == 1:
        results = [0, 1, 0, 0, 0]
    elif label_raw == 2:
        results = [0, 0, 1, 0, 0]
    elif label_raw == 3:
        results = [0, 0, 0, 1, 0]
    else:
        results = [0,0, 0, 0, 1]
    return results

# CelebA
def transformLabel_CelebA(label_raw):
    if label_raw == 0: ##
        results = [1, 0]
    else:
        results = [0, 1]
    return results

Extractor = tf.keras.applications.EfficientNetB7(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(224, 224, 3))
Extractor.layers[0].trainable = False

connection_layer = models.Sequential([Extractor,
                            layers.GlobalAveragePooling2D(),
                            layers.BatchNormalization(),
                            layers.Dropout(0.3),
                            layers.Activation('relu')
                            ], name = 'EfficientNet')

############通过txt文件进行数据读取，txt存储有图片的路径和类别 
def GetFeature(filePath, resize_format=(224, 224), resize_interpolation=cv2.INTER_LANCZOS4): 
    tmpData = []
    tmpLabel = []
    with open(filePath) as txtData:
        lines = txtData.readlines()
        for line in lines:
            file, label = line.strip().split() 
            ####将string转换为int
            label=int(label)
            tmpLabel.append(transformLabel(label))
            fileName = file
            img = cv2.imread(fileName, 1)
            img_formated = cv2.resize(img, resize_format, interpolation=resize_interpolation)
            img_formated = np.expand_dims(img_formated, axis=0)
            img_flat=connection_layer.predict(img_formated)
            img_flat = img_flat.ravel()
            tmpData.append(img_flat)
    return np.double(tmpData), np.double(tmpLabel)####返回样本和标签

# BLS
def ER_BLSNet(filePath_train,filePath_test,s,c,N1,N2,N3):
    
    train_x, train_y = GetFeature(filePath_train)
    test_x,test_y = GetFeature(filePath_test)

    L = 0
    train_x = preprocessing.scale(train_x,axis = 1)# ,with_mean = '0') #处理数据 
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0],1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0],N2*N1])
    Beta1OfEachWindow = []
    distOfMaxAndMin = []
    minOfEachWindow = []
    ymin = 0
    ymax = 1

    time_start=time.time()#计时开始
    for i in range(N2):
        random.seed(i)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1]+1,N1)-1;
#        WeightOfEachWindow([],[],i) = weightOfEachWindow; #存储每个窗口的权重系数
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias,weightOfEachWindow)
        #压缩每个窗口特征到[-1，1]
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        #通过稀疏化计算映射层每个窗口内的最终权重
        betaOfEachWindow  =  sparse_bls(FeatureOfEachWindowAfterPreprocess,FeatureOfInputDataWithBias).T
        #存储每个窗口的系数化权重
        Beta1OfEachWindow.append(betaOfEachWindow)
        #每个窗口的输出 T1
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias,betaOfEachWindow)
#        print('Feature nodes in window: max:',np.max(outputOfEachWindow),'min:',np.min(outputOfEachWindow))
        distOfMaxAndMin.append(np.max(outputOfEachWindow,axis =0) - np.min(outputOfEachWindow,axis=0))
        minOfEachWindow.append(np.min(outputOfEachWindow,axis = 0))
        outputOfEachWindow = (outputOfEachWindow-minOfEachWindow[i])/distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:,N1*i:N1*(i+1)] = outputOfEachWindow
        del outputOfEachWindow 
        del FeatureOfEachWindow 
        del weightOfEachWindow 
        
    #生成强化层
    #以下为映射层输出加偏置（强化层输入）
    InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0],1))])
    #生成强化层权重
    if N1*N2>=N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3))-1
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3).T-1).T
    
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias,weightOfEnhanceLayer)
    parameterOfShrink = s/np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = kerf(tempOfOutputOfEnhanceLayer * parameterOfShrink)####激活函数tansig relu pinv sigmoid
    
    #生成最终输入
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer,OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayer,c)
    OutputWeight = np.dot(pinvOfInput,train_y) #全局违逆
    time_end=time.time() #训练完成
    trainTime = time_end - time_start
    
    #训练输出
    OutputOfTrain = np.dot(InputOfOutputLayer,OutputWeight)###输出
    train_Prelabel=OutputOfTrain.argmax(axis=1)

    trainAcc = show_accuracy(OutputOfTrain,train_y) 
    trainPrec = show_precision(OutputOfTrain,train_y)  
    trainRecall = show_recall(OutputOfTrain,train_y) 
    trainF1_score = show_f1score(OutputOfTrain,train_y)  
    trainAUC = show_auc(OutputOfTrain,train_y)  

    print("trainTime is:", trainTime)
    print('Training accuracy is' ,trainAcc*100)
    print('Training precision is' ,trainPrec*100)
    print('Training Recall is' ,trainRecall*100)
    print('Training F1_score is' ,trainF1_score*100)
    print('Training AUC is' ,trainAUC*100)

    #测试过程
    test_x = preprocessing.scale(test_x,axis = 1)#,with_mean = True,with_std = True) #处理数据 x = (x-mean(x))/std(x) x属于[-1，1]
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0],1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0],N2*N1])
    time_start=time.time()#测试计时开始
#  映射层
    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest,Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:,N1*i:N1*(i+1)] =(ymax-ymin)*(outputOfEachWindowTest-minOfEachWindow[i])/distOfMaxAndMin[i]-ymin
#  强化层
    InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0],1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest,weightOfEnhanceLayer)
#  强化层输出
    OutputOfEnhanceLayerTest = kerf(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)####激活函数 tansig、relu pinv sigmoid
#  最终层输入
    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest,OutputOfEnhanceLayerTest])
#  最终测试输出
    OutputOfTest = np.dot(InputOfOutputLayerTest,OutputWeight)
    # test_Prelabel=OutputOfTest.argmax(axis=1)##返回最大值的下标
    time_end=time.time() #test完成
    testTime = time_end - time_start

    testAcc = show_accuracy(OutputOfTest,test_y)
    testPrec = show_precision(OutputOfTest,test_y)
    testRecall = show_recall(OutputOfTest,test_y)
    testF1_score = show_f1score(OutputOfTest,test_y)
    testAUC = show_auc(OutputOfTest,test_y)

    print("testTime is", testTime)
    print('Testing accuracy is' ,testAcc * 100)
    print('Testing precision is' ,testPrec * 100)
    print('Testing recall is' ,testRecall * 100)
    print('Testing f1_score is' ,testF1_score * 100)
    print('Testing AUC is' ,testAUC * 100)

    return testAcc
    # return testAcc, testPrec, testRecall, testF1_score, testAUC

#%%%%%%%%%%%%%%%%%%%%%%%%
'''
增加强化层节点版---BLS

参数列表：
s------收敛系数
c------正则化系数
N1–---映射层每个窗口内节点数
N2–---映射层窗口数
N3–---强化层节点数
l------步数
M------步长
'''
def ER_BLS_AddEnhanceNodes(filePath_train,filePath_test,s,c,N1,N2,N3,L,M):
    #生成映射层
    '''
    两个参数最重要，1）y;2)Beta1OfEachWindow
    '''
    train_x, train_y = GetFeature(filePath_train)
    test_x,test_y = GetFeature(filePath_test)

    u = 0
    ymax = 1 #数据收缩上限
    ymin = 0 #数据收缩下限
    train_x = preprocessing.scale(train_x,axis = 1) #处理数据 
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0],1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0],N2*N1])
#    Beta1OfEachWindow = np.zeros([N2,train_x.shape[1]+1,N1])
    distOfMaxAndMin = []
    minOfEachWindow = []
    train_acc = np.zeros([1,L+1])
    test_acc = np.zeros([1,L+1])
    train_time = np.zeros([1,L+1])
    test_time = np.zeros([1,L+1])
    time_start=time.time()#计时开始
    Beta1OfEachWindow = []
    for i in range(N2):
        random.seed(i+u)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1]+1,N1)-1; #生成每个窗口的权重系数，最后一行为偏差
#        WeightOfEachWindow([],[],i) = weightOfEachWindow; #存储每个窗口的权重系数
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias,weightOfEachWindow) #生成每个窗口的特征
        #压缩每个窗口特征到[-1，1]
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        #通过稀疏化计算映射层每个窗口内的最终权重
        betaOfEachWindow  =  sparse_bls(FeatureOfEachWindowAfterPreprocess,FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfEachWindow)
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias,betaOfEachWindow)
        distOfMaxAndMin.append( np.max(outputOfEachWindow,axis =0) - np.min(outputOfEachWindow,axis =0))
        minOfEachWindow.append(np.min(outputOfEachWindow,axis =0))
        outputOfEachWindow = (outputOfEachWindow-minOfEachWindow[i])/distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:,N1*i:N1*(i+1)] = outputOfEachWindow
        del outputOfEachWindow 
        del FeatureOfEachWindow 
        del weightOfEachWindow 
        
    #生成强化层
    #以下为映射层输出加偏置（强化层输入）
    InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0],1))])
    #生成强化层权重
    if N1*N2>=N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3)-1)
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3).T-1).T
    
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias,weightOfEnhanceLayer)
    parameterOfShrink = s/np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = kerf(tempOfOutputOfEnhanceLayer * parameterOfShrink)
    
    #生成最终输入
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer,OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayer,c)
    OutputWeight = pinvOfInput.dot(train_y) #全局违逆
    time_end=time.time() #训练完成
    trainTime = time_end - time_start
    
    #训练输出
    OutputOfTrain = np.dot(InputOfOutputLayer,OutputWeight)

    trainAcc = show_accuracy(OutputOfTrain,train_y) 
    trainPrec = show_precision(OutputOfTrain,train_y)  
    trainRecall = show_recall(OutputOfTrain,train_y) 
    trainF1_score = show_f1score(OutputOfTrain,train_y)  
    trainAUC = show_auc(OutputOfTrain,train_y)  

    print("trainTime is:", trainTime)
    print('Training accurate is' ,trainAcc*100)
    print('Training precision is' ,trainPrec*100)
    print('Training Recall is' ,trainRecall*100)
    print('Training F1_score is' ,trainF1_score*100)
    print('Training AUC is' ,trainAUC*100)

    #测试过程
    test_x = preprocessing.scale(test_x,axis = 1) #处理数据 x = (x-mean(x))/std(x) x属于[-1，1]
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0],1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0],N2*N1])
    time_start=time.time()#测试计时开始
#  映射层
    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest,Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:,N1*i:N1*(i+1)] = (ymax - ymin)*(outputOfEachWindowTest-minOfEachWindow[i])/distOfMaxAndMin[i]-ymin
#  强化层
    InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0],1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest,weightOfEnhanceLayer)
#  强化层输出
    OutputOfEnhanceLayerTest = kerf(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)    
#  最终层输入
    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest,OutputOfEnhanceLayerTest])
#  最终测试输出   
    OutputOfTest = np.dot(InputOfOutputLayerTest,OutputWeight)

    time_end=time.time() #test完成
    testTime = time_end - time_start

    testAcc = show_accuracy(OutputOfTest,test_y)
    testPrec = show_precision(OutputOfTest,test_y)
    testRecall = show_recall(OutputOfTest,test_y)
    testF1_score = show_f1score(OutputOfTest,test_y)
    testAUC = show_auc(OutputOfTest,test_y)

    print("testTime is", testTime)
    print('Testing accuracy is' ,testAcc * 100)
    print('Testing precision is' ,testPrec * 100)
    print('Testing recall is' ,testRecall * 100)
    print('Testing f1_score is' ,testF1_score * 100)
    print('Testing AUC is' ,testAUC * 100)

    print('Testing accuracy is' ,testAcc * 100)
    print('Testing precision is' ,testPrec * 100)
    print('Testing recall is' ,testRecall * 100)
    print('Testing f1_score is' ,testF1_score * 100)
    print('Testing AUC is' ,testAUC * 100)
    print('Testing time is ',testTime,'s')

    '''
        增量增加强化节点
    '''
    parameterOfShrinkAdd = []
    for e in list(range(L)):
        time_start=time.time()
        if N1*N2>= M : 
            random.seed(e)
            weightOfEnhanceLayerAdd = LA.orth(2 * random.randn(N2*N1+1,M)-1)
        else :
            random.seed(e)
            weightOfEnhanceLayerAdd = LA.orth(2 * random.randn(N2*N1+1,M).T-1).T
        tempOfOutputOfEnhanceLayerAdd = np.dot(InputOfEnhanceLayerWithBias,weightOfEnhanceLayerAdd)
        parameterOfShrinkAdd.append(s/np.max(tempOfOutputOfEnhanceLayerAdd))
        OutputOfEnhanceLayerAdd = kerf(tempOfOutputOfEnhanceLayerAdd*parameterOfShrinkAdd[e])
        tempOfLastLayerInput = np.hstack([InputOfOutputLayer,OutputOfEnhanceLayerAdd])
        
        D = pinvOfInput.dot(OutputOfEnhanceLayerAdd)
        C = OutputOfEnhanceLayerAdd - InputOfOutputLayer.dot(D)
        if C.all() == 0:
            w = D.shape[1]
            B = np.mat(np.eye(w) - np.dot(D.T,D)).I.dot(np.dot(D.T,pinvOfInput))
        else:
            B = pinv(C,c)
        pinvOfInput = np.vstack([(pinvOfInput - D.dot(B)),B])
        OutputWeightEnd = pinvOfInput.dot(train_y)
        InputOfOutputLayer = tempOfLastLayerInput
        Training_time = time.time() - time_start
        train_time[0][e+1] = Training_time
        OutputOfTrain1 = InputOfOutputLayer.dot(OutputWeightEnd)

        #增量增加节点的 测试过程
        time_start = time.time()
        OutputOfEnhanceLayerAddTest = kerf(InputOfEnhanceLayerWithBiasTest.dot(weightOfEnhanceLayerAdd) * parameterOfShrinkAdd[e]);
        InputOfOutputLayerTest=np.hstack([InputOfOutputLayerTest, OutputOfEnhanceLayerAddTest])

        OutputOfTest1 = InputOfOutputLayerTest.dot(OutputWeightEnd)

        testAcc = show_accuracy(OutputOfTest1,test_y)
        testPrec = show_precision(OutputOfTest1,test_y)
        testRecall = show_recall(OutputOfTest1,test_y)
        testF1_score = show_f1score(OutputOfTest1,test_y)
        testAUC = show_auc(OutputOfTest1,test_y)

        print('Testing accuracy is' ,testAcc * 100)
        print('Testing precision is' ,testPrec * 100)
        print('Testing recall is' ,testRecall * 100)
        print('Testing f1_score is' ,testF1_score * 100)
        print('Testing AUC is' ,testAUC * 100)
        Test_time = time.time() - time_start
        test_time[0][e+1] = Test_time     
    return test_acc,test_time,train_acc,train_time
'''
增加强化层节点版---BLS

参数列表：
s------收敛系数
c------正则化系数
N1–---映射层每个窗口内节点数
N2–---映射层窗口数
N3–---强化层节点数
L------步数

M1–---增加映射节点数
M2–---与增加映射节点对应的强化节点数
M3–---新增加的强化节点
'''
#%%%%%%%%%%%%%%%%
def ER_BLS_AddFeatureEnhanceNodes(filePath_train, filePath_test, s, c, N1, N2, N3, L, M1, M2, M3):
    
    #生成映射层
    '''
    两个参数最重要，1）y;2)Beta1OfEachWindow
    '''
    train_x, train_y = GetFeature(filePath_train)
    test_x,test_y = GetFeature(filePath_test)    
    u = 0
    ymax = 1
    ymin = 0
    train_x = preprocessing.scale(train_x,axis = 1) 
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0],1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0],N2*N1])
#    Beta1OfEachWindow = np.zeros([N2,train_x.shape[1]+1,N1])###############################
#    Beta1OfEachWindow2 = np.zeros([L,train_x.shape[1]+1,M1])
    Beta1OfEachWindow = list()
    distOfMaxAndMin = []
    minOfEachWindow = []
    train_acc = np.zeros([1,L+1])
    test_acc = np.zeros([1,L+1])
    train_time = np.zeros([1,L+1])
    test_time = np.zeros([1,L+1])
    time_start=time.time()#计时开始
    for i in range(N2):
        random.seed(i+u)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1]+1,N1)-1; #生成每个窗口的权重系数，最后一行为偏差
#        WeightOfEachWindow([],[],i) = weightOfEachWindow; #存储每个窗口的权重系数
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias,weightOfEachWindow) #生成每个窗口的特征
        #压缩每个窗口特征到[-1，1]
        scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        #通过稀疏化计算映射层每个窗口内的最终权重
        betaOfEachWindow  =  sparse_bls(FeatureOfEachWindowAfterPreprocess,FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfEachWindow)
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias,betaOfEachWindow)
        distOfMaxAndMin.append(np.max(outputOfEachWindow,axis = 0) - np.min(outputOfEachWindow,axis = 0))
        minOfEachWindow.append(np.mean(outputOfEachWindow,axis = 0))
        outputOfEachWindow = (outputOfEachWindow-minOfEachWindow[i])/distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:,N1*i:N1*(i+1)] = outputOfEachWindow
        del outputOfEachWindow 
        del FeatureOfEachWindow 
        del weightOfEachWindow 
        
    #生成强化层
    #以下为映射层输出加偏置（强化层输入）
    InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0],1))])
    #生成强化层权重
    if N1*N2>=N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3)-1)
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3).T-1).T
    
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias,weightOfEnhanceLayer)
    parameterOfShrink = s/np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = kerf(tempOfOutputOfEnhanceLayer * parameterOfShrink)
    
    #生成最终输入
    InputOfOutputLayerTrain = np.hstack([OutputOfFeatureMappingLayer,OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayerTrain,c)
    OutputWeight =pinvOfInput.dot(train_y) #全局违逆
    time_end=time.time() #训练完成
    trainTime = time_end - time_start
    
    #训练输出
    OutputOfTrain = np.dot(InputOfOutputLayerTrain,OutputWeight)

    trainAcc = show_accuracy(OutputOfTrain,train_y) 
    trainPrec = show_precision(OutputOfTrain,train_y)  
    trainRecall = show_recall(OutputOfTrain,train_y) 
    trainF1_score = show_f1score(OutputOfTrain,train_y)  
    trainAUC = show_auc(OutputOfTrain,train_y)  

    print("trainTime is:", trainTime)
    print('Training accuracy is' ,trainAcc*100)
    print('Training precision is' ,trainPrec*100)
    print('Training Recall is' ,trainRecall*100)
    print('Training F1_score is' ,trainF1_score*100)
    print('Training AUC is' ,trainAUC*100)

    #测试过程
    test_x = preprocessing.scale(test_x,axis = 1) #处理数据 x = (x-mean(x))/std(x) x属于[-1，1]
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0],1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0],N2*N1])
    time_start=time.time()#测试计时开始
#  映射层
    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest,Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:,N1*i:N1*(i+1)] = (ymax-ymin)*(outputOfEachWindowTest-minOfEachWindow[i])/distOfMaxAndMin[i] - ymin
#  强化层
    InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0],1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest,weightOfEnhanceLayer)
#  强化层输出
    OutputOfEnhanceLayerTest = kerf(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)    
#  最终层输入
    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest,OutputOfEnhanceLayerTest])
#  最终测试输出   
    OutputOfTest = np.dot(InputOfOutputLayerTest,OutputWeight)

    time_end=time.time() #训练完成
    testTime = time_end - time_start

    testAcc = show_accuracy(OutputOfTest,test_y)
    testPrec = show_precision(OutputOfTest,test_y)
    testRecall = show_recall(OutputOfTest,test_y)
    testF1_score = show_f1score(OutputOfTest,test_y)
    testAUC = show_auc(OutputOfTest,test_y)

    print('Testing accuracy is' ,testAcc * 100)
    print('Testing precision is' ,testPrec * 100)
    print('Testing recall is' ,testRecall * 100)
    print('Testing f1_score is' ,testF1_score * 100)
    print('Testing AUC is' ,testAUC * 100)
    print('Testing time is ',testTime,'s')
    '''
        增加Mapping 和 强化节点
    '''
    WeightOfNewFeature2 = list()
    WeightOfNewFeature3 = list()
    for e in list(range(L)):
        time_start = time.time()
        random.seed(e+N2+u)
        weightOfNewMapping = 2 * random.random([train_x.shape[1]+1,M1]) - 1
        NewMappingOutput = FeatureOfInputDataWithBias.dot(weightOfNewMapping)
#        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias,weightOfEachWindow) #生成每个窗口的特征
        #压缩每个窗口特征到[-1，1]
        scaler2 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(NewMappingOutput)
        FeatureOfEachWindowAfterPreprocess = scaler2.transform(NewMappingOutput)
        betaOfNewWindow  =  sparse_bls(FeatureOfEachWindowAfterPreprocess,FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfNewWindow)
   
        TempOfFeatureOutput = FeatureOfInputDataWithBias.dot(betaOfNewWindow)
        distOfMaxAndMin.append( np.max(TempOfFeatureOutput,axis = 0) - np.min(TempOfFeatureOutput,axis = 0))
        minOfEachWindow.append(np.mean(TempOfFeatureOutput,axis = 0))
        outputOfNewWindow = (TempOfFeatureOutput-minOfEachWindow[N2+e])/distOfMaxAndMin[N2+e]
        #新的映射层整体输出
        OutputOfFeatureMappingLayer = np.hstack([OutputOfFeatureMappingLayer,outputOfNewWindow])
        # 新增加映射窗口的输出带偏置
        NewInputOfEnhanceLayerWithBias = np.hstack([outputOfNewWindow, 0.1 * np.ones((outputOfNewWindow.shape[0],1))])
        #新映射窗口对应的强化层节点，M2列
        if M1 >= M2:
            random.seed(67797325)
            RelateEnhanceWeightOfNewFeatureNodes = LA.orth(2*random.random([M1+1,M2])-1)
        else:
            random.seed(67797325)
            RelateEnhanceWeightOfNewFeatureNodes = LA.orth(2*random.random([M1+1,M2]).T-1).T  
        WeightOfNewFeature2.append(RelateEnhanceWeightOfNewFeatureNodes)    
        tempOfNewFeatureEhanceNodes = NewInputOfEnhanceLayerWithBias.dot(RelateEnhanceWeightOfNewFeatureNodes)
        parameter1 = s/np.max(tempOfNewFeatureEhanceNodes)
        #与新增的Feature Mapping 节点对应的强化节点输出
        outputOfNewFeatureEhanceNodes = kerf(tempOfNewFeatureEhanceNodes * parameter1)
        if N2*N1+e*M1>=M3:
            random.seed(67797325+e)
            weightOfNewEnhanceNodes = LA.orth(2 * random.randn(N2*N1+(e+1)*M1+1,M3) - 1)
        else:
            random.seed(67797325+e)
            weightOfNewEnhanceNodes = LA.orth(2 * random.randn(N2*N1+(e+1)*M1+1,M3).T-1).T
        WeightOfNewFeature3.append(weightOfNewEnhanceNodes)
        # 整体映射层输出带偏置
        InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0],1))])

        tempOfNewEnhanceNodes = InputOfEnhanceLayerWithBias.dot(weightOfNewEnhanceNodes)
        parameter2 = s/np.max(tempOfNewEnhanceNodes)
        OutputOfNewEnhanceNodes = kerf(tempOfNewEnhanceNodes * parameter2);
        OutputOfTotalNewAddNodes = np.hstack([outputOfNewWindow,outputOfNewFeatureEhanceNodes,OutputOfNewEnhanceNodes])
        tempOfInputOfLastLayes = np.hstack([InputOfOutputLayerTrain,OutputOfTotalNewAddNodes])
        D = pinvOfInput.dot(OutputOfTotalNewAddNodes)
        C = OutputOfTotalNewAddNodes - InputOfOutputLayerTrain.dot(D)
        
        if C.all() == 0:
            w = D.shape[1]
            B = (np.eye(w)- D.T.dot(D)).I.dot(D.T.dot(pinvOfInput))
        else:
            B = pinv(C,c)
        pinvOfInput = np.vstack([(pinvOfInput - D.dot(B)),B])
        OutputWeight = pinvOfInput.dot(train_y)        
        InputOfOutputLayerTrain = tempOfInputOfLastLayes
        
        time_end = time.time()
        Train_time = time_end - time_start
        train_time[0][e+1] = Train_time
        predictLabel = InputOfOutputLayerTrain.dot(OutputWeight)
        # TrainingAccuracy = show_accuracy(predictLabel,train_y)
    
        # 测试过程
        #先生成新映射窗口输出
        time_start = time.time() 
        WeightOfNewMapping =  Beta1OfEachWindow[N2+e]
        outputOfNewWindowTest = FeatureOfInputDataWithBiasTest.dot(WeightOfNewMapping )
         #TT1
        outputOfNewWindowTest = (ymax - ymin)*(outputOfNewWindowTest-minOfEachWindow[N2+e])/distOfMaxAndMin[N2+e] - ymin
        ## 整体映射层输出
        OutputOfFeatureMappingLayerTest = np.hstack([OutputOfFeatureMappingLayerTest,outputOfNewWindowTest])
        # HH2
        InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest,0.1*np.ones([OutputOfFeatureMappingLayerTest.shape[0],1])])
        # hh2
        NewInputOfEnhanceLayerWithBiasTest = np.hstack([outputOfNewWindowTest,0.1*np.ones([outputOfNewWindowTest.shape[0],1])])
        weightOfRelateNewEnhanceNodes = WeightOfNewFeature2[e]
        #tt22
        OutputOfRelateEnhanceNodes = kerf(NewInputOfEnhanceLayerWithBiasTest.dot(weightOfRelateNewEnhanceNodes) * parameter1)
        weightOfNewEnhanceNodes = WeightOfNewFeature3[e]
        # tt2
        OutputOfNewEnhanceNodes = kerf(InputOfEnhanceLayerWithBiasTest.dot(weightOfNewEnhanceNodes)*parameter2)   
        InputOfOutputLayerTest = np.hstack([InputOfOutputLayerTest,outputOfNewWindowTest,OutputOfRelateEnhanceNodes,OutputOfNewEnhanceNodes])
        predictLabel = InputOfOutputLayerTest.dot(OutputWeight)

        time_end = time.time()
        Testing_time= time_end - time_start

        testAcc = show_accuracy(predictLabel,test_y)
        testPrec = show_precision(predictLabel,test_y)
        testRecall = show_recall(predictLabel,test_y)
        testF1_score = show_f1score(predictLabel,test_y)
        testAUC = show_auc(predictLabel,test_y)

        print('Testing time is ',Testing_time,'s')
        print('Testing accuracy is' ,testAcc * 100)
        print('Testing precision is' ,testPrec * 100)
        print('Testing recall is' ,testRecall * 100)
        print('Testing f1_score is' ,testF1_score * 100)
        print('Testing AUC is' ,testAUC * 100)


    return test_acc,test_time,train_acc,train_time
'''
#############################增加训练数据
'''
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def ER_BLS_AddNewData(filePath_train, filePath_test, s, C, N1, N2, N3, l, m):
    '''
    %Incremental Learning Process of the proposed broad learning system: for
    %increment of input patterns
    %Input: 
    %---train_x,test_x : the training data and learning data in the begining of###刚开始时候的输入数据
    %the incremental learning##增量学习
    %---train_y,test_y : the label
    %---train_yf,train_xf: the whold training samples of the learning system训练过程中所有的训练样本
    %---We: the randomly generated coefficients of feature nodes随机产生的特征节点系数
    %---wh:the randomly generated coefficients of enhancement nodes随机产生的增强节点系数
    %----s: the shrinkage parameter for enhancement nodes增强节点的收缩系数
    %----C: the regularization parameter for sparse regualarization正则化系数
    %----N1: the number of feature nodes  per window每一个窗口内的特征参数
    %----N2: the number of windows of feature nodes窗口数
    %----N3: the number of enhancements nodes增强节点数
    % ---m:number of added input patterns per increment step每个增量步骤添加的输入样本数量
    % ---l: steps of incremental learning###增量步骤

    %output:
    %---------Testing_time1:Accumulative Testing Times
    %---------Training_time1:Accumulative Training Time
    '''
    # bls_train_input(traindata[0:10000,:],trainlabel[0:10000,:],traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,l,m)
    train_x, train_y = GetFeature(filePath_train)
    test_x, test_y = GetFeature(filePath_test)
    train_xf = train_x[0:84533,:]
    train_yf = train_y[0:84533,:]

    u = 0 #random seed
    ymin = 0
    ymax = 1 
    train_err = np.zeros([1,l+1])
    test_err = np.zeros([1,l+1])
    train_time = np.zeros([1,l+1])
    test_time = np.zeros([1,l+1])
    minOfEachWindow = []
    distMaxAndMin = []
    beta11 = list()
    Wh = list()
    '''
    feature nodes
    '''
    time_start = time.time()
    train_x = preprocessing.scale(train_x, axis = 1) ###训练数据进行预处理
    H1 = np.hstack([train_x, .1 * np.ones([train_x.shape[0], 1])])###产生一个随机权重矩阵，后边添加一列（偏置）为了便于计算
    y = np.zeros([train_x.shape[0], N2*N1]);####
    for i in range(N2):
        random.seed(i+u)
        we= 2 * random.randn(train_x.shape[1]+1,N1)-1
        A1 = H1.dot(we)
        scaler2 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(A1)
        A1 = scaler2.transform(A1)
        beta1 =  sparse_bls(A1,H1).T
        beta11.append(beta1)
        T1 = H1.dot(beta1)
        minOfEachWindow.append(T1.min(axis = 0))
        distMaxAndMin.append( T1.max(axis = 0) - T1.min(axis = 0))
        T1 = (T1 - minOfEachWindow[i])/distMaxAndMin[i]
        y[:,N1*i:N1*(i+1)] = T1
        
    '''
    enhancement nodes
    '''
    H2 = np.hstack([y,0.1 * np.ones([y.shape[0],1])])
    if N1*N2>=N3 :
        random.seed(67797325)
        wh = LA.orth(2 * random.randn(N2*N1+1,N3)-1)
    else:
        random.seed(67797325)
        wh = LA.orth(2 * random.randn(N2*N1+1,N3).T-1).T                 
    Wh.append(wh)
    T2 = H2.dot(wh)
    parameter = s/np.max(T2)
    T2 = kerf(T2 * parameter);  ##########tansig   sigmoid
    T3 = np.hstack([y,T2])
    beta = pinv(T3,C)
    beta2 = beta.dot(train_y)
    Training_time = time.time() - time_start
    train_time[0][0] =Training_time
    print('Training has been finished!')
    print('The Total Training Time is : ', Training_time, 's' )
    xx = T3.dot(beta2)
    '''
    Testing Process
    '''
    time_start = time.time()
    test_x = preprocessing.scale(test_x,axis = 1) 
    HH1 = np.hstack([test_x, 0.1 * np.ones([test_x.shape[0],1])])
    yy1=np.zeros([test_x.shape[0],N2*N1])
    for i in range(N2):
        beta1 = beta11[i]
        TT1 = HH1.dot(beta1)
        TT1 = (ymax - ymin)*(TT1 - minOfEachWindow[i])/distMaxAndMin[i] - ymin
        yy1[:,N1*i:N1*(i+1)]= TT1
    HH2 = np.hstack([yy1, 0.1 * np.ones([yy1.shape[0],1])]); 
    TT2 = kerf(HH2.dot(wh) * parameter)  #######tansig  sigmoid
    TT3 = np.hstack([yy1,TT2])

    '''
    testing accuracy
    '''
    x = TT3.dot( beta2)
    Testing_time = time.time()- time_start
    test_time[0][0] = Testing_time
    print('Testing has been finished!')
    print('The Total Testing Time is : ', Testing_time, 's' )   

    testAcc = show_accuracy(x,test_y)
    testPrec = show_precision(x,test_y)
    testRecall = show_recall(x,test_y)
    testF1_score = show_f1score(x,test_y)
    testAUC = show_auc(x,test_y)

    print('Testing accuracy is' ,testAcc * 100)
    print('Testing precision is' ,testPrec * 100)
    print('Testing recall is' ,testRecall * 100)
    print('Testing f1_score is' ,testF1_score * 100)
    print('Testing AUC is' ,testAUC * 100)
 
    '''
    incremental training steps,前边都是比较常规的计算，后边是数据增加时候的计算内容
    '''
    for e in range(l):
        time_start = time.time()
        '''
   WARNING: If data comes from a single dataset, the following 'train_xx' and 'train_y1' should be reset!
        '''
        train_xx = preprocessing.scale(train_xf[(84533+(e)*m):(84533+(e+1)*m),:],axis = 1) ####[(10000+(e)*m):(10000+(e+1)*m),:]  584
        train_y1 = train_yf[0:84533+(e+1)*m,:]####[0:10000+(e+1)*m,:]

        Hx1 = np.hstack([train_xx, 0.1 * np.ones([train_xx.shape[0],1])])
        yx = np.zeros([train_xx.shape[0],N1*N2])
        for i in range(N2):
            beta1 = beta11[i]
            Tx1 = Hx1.dot(beta1)
            Tx1 = (ymax - ymin)*(Tx1 - minOfEachWindow[i])/distMaxAndMin[i] - ymin
            yx[:,N1*i:N1*(i+1)]= Tx1
                                       
        Hx2 = np.hstack([yx, 0.1 * np.ones([yx.shape[0],1])]);
        wh = Wh[0]
        t2 = kerf(Hx2.dot(wh) * parameter);#########tansig  sigmoid
        t2 = np.hstack([yx, t2])
        betat = pinv(t2,C)
        beta = np.hstack([beta, betat])
        beta2 = beta.dot(train_y1)
        T3 = np.vstack([T3,t2])
        Training_time= time.time()- time_start
        print(Training_time)
        train_time[0][e+1] = Training_time
        xx = T3.dot( beta2)
        TrainingAccuracy = show_accuracy(xx,train_y1)
        train_err[0][e+1] = TrainingAccuracy
        '''
        incremental testing steps
        '''
        time_start = time.time()
        x = TT3.dot(beta2)

        testAcc = show_accuracy(x,test_y)
        testPrec = show_precision(x,test_y)
        testRecall = show_recall(x,test_y)
        testF1_score = show_f1score(x,test_y)
        testAUC = show_auc(x,test_y)

        Testing_time = time.time() - time_start
        test_time[0][e+1] = Testing_time
        print('Testing has been finished!')
        print('The Total Testing Time is : ', Testing_time, 's' )

        print('Testing accuracy is' ,testAcc * 100)
        print('Testing precision is' ,testPrec * 100)
        print('Testing recall is' ,testRecall * 100)
        print('Testing f1_score is' ,testF1_score * 100)
        print('Testing AUC is' ,testAUC * 100)

    # return test_err,test_time,train_err,train_time