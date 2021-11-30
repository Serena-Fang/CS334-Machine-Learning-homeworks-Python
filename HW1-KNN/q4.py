import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
import knn


def standard_scale(xTrain, xTest):
    arrayTrain = np.array(xTrain)
    xTrain = preprocessing.scale(arrayTrain)
    arrayTest = np.array(xTest)
    xTest = preprocessing.scale(arrayTest)
    return xTrain, xTest


def minmax_range(xTrain, xTest):
    scaler = preprocessing.MinMaxScaler()
    train = np.array(xTrain)
    test = np.array(xTest)
    xTrain = scaler.fit_transform(train)
    xTest = scaler.fit_transform(test)
    return xTrain, xTest


def add_irr_feature(xTrain, xTest):
    n_train = xTrain.shape[0]
    n_test = xTest.shape[0]
    xTrain['f1'] = np.random.normal(0, 1, n_train)
    xTrain['f2'] = np.random.normal(0, 1, n_train)
    xTest['f1'] = np.random.normal(0, 1, n_test)
    xTest['f2'] = np.random.normal(0, 1, n_test)
    return xTrain, xTest


def knn_train_test(k, xTrain, yTrain, xTest, yTest):
    model = knn.Knn(k)
    model.train(xTrain, yTrain['label'])
    # predict the test dataset
    yHatTest = model.predict(xTest)
    return knn.accuracy(yHatTest, yTest['label'])
    

def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
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

    # no preprocessing
    acc1 = knn_train_test(args.k, xTrain, yTrain, xTest, yTest)
    print("Test Acc (no-preprocessing):", acc1)
    # preprocess the data using standardization scaling
    xTrainStd, xTestStd = standard_scale(xTrain, xTest)
    acc2 = knn_train_test(args.k, xTrainStd, yTrain, xTestStd, yTest)
    print("Test Acc (standard scale):", acc2)
    # preprocess the data using min max scaling
    xTrainMM, xTestMM = minmax_range(xTrain, xTest)
    acc3 = knn_train_test(args.k, xTrainMM, yTrain, xTestMM, yTest)
    print("Test Acc (min max scale):", acc3)
    # add irrelevant features
    xTrainIrr, yTrainIrr = add_irr_feature(xTrain, xTest)
    acc4 = knn_train_test(args.k, xTrainIrr, yTrain, yTrainIrr, yTest)
    print("Test Acc (with irrelevant feature):", acc4)

    k_range = range(1, 26)
    no_preprocessing_scores = []
    standard_scale_scores = []
    min_max_scores = []
    irrelevant_scores = []

    for k in k_range:
        acc1 = knn_train_test(args.k, xTrain, yTrain, xTest, yTest)
        xTrainStd, xTestStd = standard_scale(xTrain, xTest)
        acc2 = knn_train_test(args.k, xTrainStd, yTrain, xTestStd, yTest)
        xTrainMM, xTestMM = minmax_range(xTrain, xTest)
        acc3 = knn_train_test(args.k, xTrainMM, yTrain, xTestMM, yTest)
        xTrainIrr, yTrainIrr = add_irr_feature(xTrain, xTest)
        acc4 = knn_train_test(args.k, xTrainIrr, yTrain, yTrainIrr, yTest)
        no_preprocessing_scores.append(acc1)
        standard_scale_scores.append(acc2)
        min_max_scores.append(acc3)
        irrelevant_scores.append(acc4)

    print(no_preprocessing_scores)
    print(standard_scale_scores)
    print(min_max_scores)
    print(irrelevant_scores)

if __name__ == "__main__":
    main()

# (d) When data is adjusted to standard scale or min max range, it is non-sensitive to the value of k.
# However, when data are non-preprocessed and with irrelevant features, the classification accuracy generally decrease as k increases.
# Figures in q4_plot.py.