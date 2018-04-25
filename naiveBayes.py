'''
Created on 23-Apr-2018

@author: Ashwin
'''

import pandas as pd
import numpy as np
import pickle
import operator
from operator import itemgetter
import sys

def naiveBayes(train, trainLabel, test, testLabel):
    d = {"1": 0}
    [totalRows, totalColumns] = train.shape
    yList = np.unique(trainLabel)
    p = priorsAndProb(d, train.toarray(), yList, trainLabel)
    print "done"
#     with open('dictFine.pkl', 'rb') as f:
#         d = pickle.load(f)
    print "predict started"
    predict(test.toarray(), d, p, yList, totalRows, testLabel)

def naiveBayesTop5ClassLabels(train, trainLabel,test):
    d = {"1": 0}
    [totalRows, totalColumns] = train.shape
    yList = np.unique(trainLabel)
    p = priorsAndProb(d, train.toarray(), yList, trainLabel)
    print "done"
#     with open('dictFine.pkl', 'rb') as f:
#         d = pickle.load(f)
    print "started"
    return predictTop5(test.toarray(), d, p, yList, totalRows)
    
def probY(testDataRow, d, p, yList, totalSize, totalCol):
    probList = {}
    #minValue = -sys.maxint - 1
    for y in range(yList.size):
        prob = p[yList[y]]
        for c in range(totalCol):
            key = str(c) +","+ str(yList[y]) +","+str(testDataRow[0,c])
            if key in d:
                prob = prob * float(d[key])
            else:
                prob = prob * 0.0000001      
        probList[yList[y]] = prob
    return probList    


def predictTop5(test, d, p, yList, totalSize):
    [totalRow, totalCol] = test.shape
    acc = 0 
    pac = 0
    for r in range(totalRow):
        print r
        probabilityY = probY(test[r:r+1,:], d, p, yList, totalSize, totalCol)
        sorted_x = sorted(probabilityY.items(), key=lambda x: x[1], reverse=True)
        testLabel = []    
        for [key,value] in sorted_x[:5]:
            testLabel.append(key)
    return testLabel


def predict(test, d, p, yList, totalSize, testLabel):
    [totalRow, totalCol] = test.shape
    acc = 0 
    pac = 0
    for r in range(totalRow):
        print r
        probabilityY = probY(test[r:r+1,:], d, p, yList, totalSize, totalCol)
        sorted_x = sorted(probabilityY.items(), key=lambda x: x[1], reverse=True)
            
        for [key,value] in sorted_x[:5]:
            if testLabel[r] == key:
                acc = acc+1   
                pac = pac+1
                if pac == 50:
                    pac = 0
                    print acc
    acc = acc/ float(totalRow)
    print acc
    return acc

def priorsAndProb(d, train, yList, trainLabel):
    prior = []
    [r,c] = train.shape
    for y in range(yList.size):
        p = findPrior(trainLabel)
        for i in range(c):
            probOfX(train, i, yList[y], d, p[yList[y]], trainLabel)
    for key in d.keys():
        d[key] = d[key] + 1 
    for y in range(yList.size):
        p[yList[y]] = p[yList[y]]/float(r)
    return p


def probOfX(train, i, y, d, p, trainLabel):
    [totalRow, totalColumn] = train.shape
    for r in range(totalRow):
        if trainLabel[r] == y:
            key = str(i) +","+ str(y) +","+str(train[r,i])
            if key in d:
                d[key] = ((d[key] * p) + 1)/p
            else:
                d[key] = 1/p
       
                
                
def findPrior(trainLabel):
    unique, counts = np.unique(trainLabel, return_counts=True)
    return dict(zip(unique, counts))



# function probOfX(X,i, y, M,p)
# yo = cell2mat(y);
# for r = 1: size(X,1)
#     trainRow = table2cell(X(r,:));
#     trainMat = cell2mat(trainRow);
#     xi = trainMat(i);
#     if strcmp(trainMat(1), yo) == 1
#         key  = sprintf('%d,%c,%c', i, yo,xi);
#         if(isKey(M,key))
#             M(key) = ((M(key) * p) + 1)/p;
#         else
#             M(key) =  1/p;
#         end
#     end
# end
# end
# 


# function p = findPrior(yValue, train)
# p = 0;
# for r = 1: size(train,1)
#     trainRow = table2cell(train(r,:));
#     if strcmp(trainRow(1), yValue) == 1
#         p = p + 1;
#     end
# end
# end 

# function [ylist,prior, totSize] = priorsAndProb(M, trainData)
# o = unique(trainData(:,1));
# ylist = table2array(o);
# totSize = size(trainData,1);
# for y = 1: size(ylist)
#     p = findPrior(ylist(y),trainData);
#     prior(y) = p/size(trainData,1);
#     for i = 2 : size(trainData,2)
#         probOfX(trainData, i, ylist(y), M,p);
#     end
# end
# end    

# 
# 
# function predictedY = predictY(testDataRow, M, prior, yo, totSize)
#     ylist = cell2mat(yo);
#     predictedY = ylist(1); maxprob = 0;
#     testRow = table2cell(testDataRow);
#     testMat = cell2mat(testRow);
#     
#     for y = 1: size(ylist)
#         prob = prior(y);
#         for c = 2: size(testDataRow,2)
#             xc = testMat(c);
#             key  = sprintf('%d,%c,%c', c, ylist(y),xc);
#             if(~isKey(M,key))
#                M(key) = 0;
#             end
#             prob = prob * M(key);
#         end
#         if maxprob < prob
#             maxprob = prob;
#             predictedY = ylist(y);
#         end
#     end
# end

# 
# function acc = predict(testData, prior, M, ylist,totSize)
#     acc = 0;
#     for r = 1: size(testData,1)
#         testRow = table2cell(testData(r,:));
#         testMat = cell2mat(testRow);
#         predictedY =  predictY(testData(r,:), M, prior, ylist, totSize);
#         if predictedY == testMat(1)
#             acc= acc+1;
#         end
#     end
#     acc = acc / size(testData,1);
# end

# function runNaiveBayes()
# M = containers.Map();
# [trainData,testData] = loadData();
# [ylist, prior, totSize] = priorsAndProb(M,trainData);
# displayMap(M);
# acc = predict(testData, prior, M, ylist,totSize);
# disp(acc);
# end