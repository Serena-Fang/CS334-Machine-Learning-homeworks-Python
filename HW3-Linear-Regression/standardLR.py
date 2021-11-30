import argparse
import numpy as np
import pandas as pd
import time

from lr import LinearRegression, file_to_numpy


class StandardLR(LinearRegression):

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        trainStats = {}
        # TODO: DO SOMETHING
        trainStat = {}
        start = time.time()
        n = xTrain.shape[0]
        int = np.ones((n, 1))
        X1 = np.append(int, xTrain, axis=1)
        self.beta = np.linalg.inv(X1.T.dot(X1)).dot(X1.T).dot(yTrain)

        end = time.time()
        trainStat['time'] = end - start

        trainStat['train-mse'] = self.mse(xTrain, yTrain)
        trainStat['test-mse'] = self.mse(xTest, yTest)
        trainStats[0] = trainStat

        return trainStats


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    model = StandardLR()
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()
