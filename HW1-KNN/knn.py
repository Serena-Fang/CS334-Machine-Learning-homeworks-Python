import argparse
# noinspection PyUnresolvedReferences
from collections import Counter
# noinspection PyUnresolvedReferences
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def euclidean_distance(x1, x2):
    distance = 0.0
    for i in range(len(x1)):
        distance += (x1[i] - x2[i]) ** 2
    return sqrt(distance)


class Knn(object):
    k = 0  # number of neighbors to use

    def __init__(self, k):

        self.k = k

    def train(self, xFeat, y):

        self.X_train = np.array(xFeat)
        self.y_train = np.array(y)

        return self

    def predict(self, xFeat):
        yHat = []  # variable to store the estimated class label
        X = np.array(xFeat)
        yHat = [self._predict(x) for x in X]
        yHat = np.array(yHat)
        return yHat

    def _predict(self, x):
        # compute distances

        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


def accuracy(yHat, yTrue):
    acc = 0
    diff = yHat - yTrue
    acc = len(diff[diff == 0]) / len(yHat)
    return acc


def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="q3xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q3yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q3xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q3yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create an instance of the model
    knn = Knn(args.k)
    knn.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain['label'])
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest['label'])
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)

    k_range = range(1, 26)
    train_scores = []
    test_scores = []
    for k in k_range:
        knn = Knn(k)
        knn.train(xTrain, yTrain['label'])
        yHatTrain = knn.predict(xTrain)
        trainAcc = accuracy(yHatTrain, yTrain['label'])
        yHatTest = knn.predict(xTest)
        testAcc = accuracy(yHatTest, yTest['label'])
        train_scores.append(trainAcc)
        test_scores.append(testAcc)


if __name__ == "__main__":
    main()

# (c) When k = 5, the train accuracy is 0.942 and the test accuracy is 0.925.
# (e) The computational complexity of the predict function is O(nd + kn).
# For each training observation, we compute the distance from it to the new observation.
# If there are n training observations with d features, the run time is O(nd).
# Then we sort the distances (size n) and loop through them to find the nearest k neighbours.
# This takes O(kn) runtime. Adding the two steps up gives the method O(nd + kn) time complexity.
# The space complexity is O(n*d) as we have to store the distance for n training observations with d features.