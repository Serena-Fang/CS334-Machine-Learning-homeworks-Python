import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from math import log
import operator
import matplotlib.pyplot as plt


class DecisionTree(object):
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion

    def __init__(self, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample

    def Entropy(self, y):
        num = len(y)
        labels = {}
        List = list(y['label'])
        for l in List:
            if l not in labels.keys():
                labels[l] = 0
            labels[l] += 1
        entropy = 0.0
        for l in labels:
            p = float(labels[l])/num
            entropy = entropy - p*log(p,2)
        return entropy

    def best_feature(self, X, y):
        best_feature = -1
        best_entropy = self.Entropy(y)
        best_InfoGain = 0.0
        best_gini = 99999.0
        for i in range(len(X.columns)):
            features = list(X[X.columns[i]])
            feat_list = set(features)

            if self.criterion == "gini":
                gini = 0.0
                for value in feat_list:
                    xFeatSub, ySub = self.splitDataset(X, y, i, value)
                    p = len(ySub) / float(len(y))
                    sub_p = len(self.splitDataset(xFeatSub, ySub, -1, '0')) / float(len(ySub))
                gini += p * (1.0 - pow(sub_p, 2) - pow(1 - sub_p, 2))
                if (gini < best_gini):
                    best_gini = gini
                    best_feature = i

            if self.criterion == 'entropy':
                entropy = 0.0
                for feat in feat_list:
                    xFeatSub, ySub = self.splitDataset(X, y, i, feat)
                    p = len(ySub) / float(len(y))
                    entropy += p * self.Entropy(ySub)
                infoGain = best_entropy - entropy
                if (infoGain > best_InfoGain):
                    best_InfoGain = infoGain
                    best_feature = i

        self.maxDepth -= 1
        return best_feature

    def splitDataset(self, X, y, i, value):
        data = pd.concat([X, y], axis = 1)
        label = data.columns[i]
        dataset = data.loc[data[label] == value]
        dataset.drop(label, axis = 1)
        Xnew = pd.concat([dataset.loc[:, list(data.columns[:i])],
                          dataset.loc[:, list(data.columns[i+1:-1])]], axis=1)
        ynew = dataset.loc[:, ['label']]
        return Xnew, ynew

    def majorityCount(self, yList):
        counter = {}
        for y in yList:
            if y not in counter.keys():
                counter[y] = 0
            counter[y] += 1
        Counter = sorted(counter.items(), key = operator.itemgetter(1), reverse=True)
        label = Counter[0][0]
        return label


    def train(self, xFeat, y):
        yList = list(y['label'])
        if yList.count(yList[0]) == len(yList):
            return yList[0]
        labels = list(xFeat.columns)
        if len(labels) == 1:
            return self.majorityCount(yList)

        best_feature = self.best_feature(xFeat, y)
        best_label = labels[best_feature]
        values = set(xFeat[best_label])
        if self.maxDepth < 0 or len(values) < self.minLeafSample:
            return self.majorityCount(yList)

        tree = {best_label: {}}
        for v in values:
            xNew, yNew = self.splitDataset(xFeat, y, best_feature, v)
            tree[best_label][v] = self.train(xNew, yNew)

        self.tree = tree

        return self.tree

    def classify(self, tree, labels, xi):
        """
        Given the feature set xFeat, predict
        what class the value will have.
        """
        string = list(tree.keys())[0]
        dic = tree[string]
        i = labels.index(string)
        classLabel = '0'
        for key in dic.keys():
            if xi[i] == key:
                if type(dic[key]).__name__ == 'dict':
                    classLabel = self.classify(dic[key], labels, xi)
                else:
                    classLabel = dic[key]
        return classLabel

    def predict(self, xFeat):
        yHat = [] # variable to store the estimated class label
        labels = list(xFeat.columns)
        for i in range(len(xFeat)):
            x = xFeat.loc[i, :]
            yHat.append(int(self.classify(self.tree, labels, x)))
        return yHat

def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    """
    Given a decision tree model, train the model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    dt : DecisionTree
        The decision tree with the model parameters
    xTrain : nd-array with shape n x d
        Training data 
    yTrain : 1d array with shape n
        Array of labels associated with training data.
    xTest : nd-array with shape m x d
        Test data 
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    # train the model
    dt.train(xTrain, yTrain)
    # predict the training dataset
    yHatTrain = dt.predict(xTrain)
    trainAcc = accuracy_score(list(yTrain['label']), yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = accuracy_score(list(yTest['label']), yHatTest)
    return trainAcc, testAcc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("md",
                        type=int,
                        help="maximum depth")
    parser.add_argument("mls",
                        type=int,
                        help="minimum leaf samples")
    parser.add_argument("--xTrain",
                        default="q4xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q4yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q4xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q4yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create an instance of the decision tree using gini
    dt1 = DecisionTree('gini', args.md, args.mls)
    trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
    print("GINI Criterion ---------------")
    print("Training Acc:", trainAcc1)
    print("Test Acc:", testAcc1)
    dt = DecisionTree('entropy', args.md, args.mls)
    trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
    print("Entropy Criterion ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)

    md_range = []
    md_train_scores = []
    md_test_scores = []
    for md in range(1,11):
        dt = DecisionTree('gini', md, args.mls)
        trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
        md_train_scores.append(trainAcc)
        md_test_scores.append(testAcc)
        md_range.append(md)

    print(md_range)
    print(md_train_scores)
    print(md_test_scores)

    mls_range = []
    mls_train_scores = []
    mls_test_scores = []
    for mls in range(1, 11):
        dt = DecisionTree('gini', args.md, mls)
        trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
        mls_train_scores.append(trainAcc)
        mls_test_scores.append(testAcc)
        mls_range.append(mls)

    print(mls_range)
    print(mls_train_scores)
    print(mls_test_scores)



if __name__ == "__main__":
    main()
