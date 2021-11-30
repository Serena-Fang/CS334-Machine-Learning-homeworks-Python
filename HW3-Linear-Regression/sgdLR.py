import argparse
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from standardLR import StandardLR

from lr import LinearRegression, file_to_numpy


class SgdLR(LinearRegression):
    lr = 1  # learning rate
    bs = 1  # batch size
    mEpoch = 1000 # maximum epoch size

    def __init__(self, lr, bs, epoch):
        self.lr = lr
        self.bs = bs
        self.mEpoch = epoch

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        trainStats = {}
        # TODO: DO SGD
        m = np.shape(xTrain)[0]
        n = np.shape(xTrain)[1]

        x1 = np.concatenate((np.ones((m,1)), xTrain), axis=1)
        self.beta = np.random.randn(n+1, 1)

        iteration = 0
        for i in range(self.mEpoch):
            i = np.arange(m)
            np.random.shuffle(i)
            X = x1[i]
            y = yTrain[i]

            batch_list = []
            for data in np.arange(start=0, stop=m, step=self.bs):
                batch_list.append((X[data: data + self.bs],
                                   y[data: data + self.bs]))

            for (xBatch, yBatch) in batch_list:
                start = time.time()
                trainStat = {}
                batch_m = np.shape(xBatch)[0]
                yPred = np.matmul(xBatch, self.beta)
                error = yPred - yBatch
                gradient = (1/batch_m) * xBatch.T.dot(error)
                self.beta = self.beta - self.lr*gradient

                trainStat['time'] = time.time() - start

                trainStat['train-mse'] = self.mse(xTrain, yTrain)
                trainStat['test-mse'] = self.mse(xTest, yTest)

                trainStats[iteration] = trainStat
                iteration += 1

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
    parser.add_argument("lr", type=float, help="learning rate")
    parser.add_argument("bs", type=int, help="batch size")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    # standardize label columns
    sc = StandardScaler()
    yTrain = sc.fit_transform(yTrain)
    yTest = sc.fit_transform(yTest)

    # setting the seed for deterministic behavior
    np.random.seed(args.seed)   
    model = SgdLR(args.lr, args.bs, args.epoch)
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)

    # Q3(b)
    indices = np.arange(np.shape(xTrain)[0])
    np.random.shuffle(indices)
    rate = int(0.4 * len(indices))
    X1, X2 = xTrain[indices[:rate]], xTrain[indices[rate + 1:]]
    y1, y2 = yTrain[indices[:rate]], yTrain[indices[rate + 1:]]

    lrs = [0.01, 0.001, 0.0001, 0.00001]
    for lr in lrs:
        model = SgdLR(lr, 1, args.epoch)
        trainStats = model.train_predict(X1, y1, X2, y2)
        test_mse = []
        r = np.arange(len(trainStats) / args.epoch, len(trainStats) + 1, len(trainStats) / args.epoch)
        for i in r:
            test_mse.append(trainStats[i - 1]['test-mse'])

        x_axis = [i for i in range(len(test_mse))]
        plt.plot(x_axis, test_mse, label='lr =' + str(lr))

    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('mse')
    plt.savefig('lr_mse.jpg')

    # Q3(c)
    best_lr = 0.001
    model = SgdLR(best_lr, 1, args.epoch)
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    train_mse = []
    test_mse = []
    for i in np.arange(len(trainStats) / args.epoch, len(trainStats) + 1, len(trainStats) / args.epoch):
        train_mse.append(trainStats[i - 1]['train-mse'])
        test_mse.append(trainStats[i - 1]['test-mse'])

    x_axis = [i for i in range(len(test_mse))]
    plt.plot(x_axis, train_mse, label='train mse')
    plt.plot(x_axis, test_mse, label='test mse')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('mse')
    plt.savefig('bset_lr_mse.png')

    # Q4(a)
    # Choose a variety of batch sizes (len(xTrain)=16770)
    bs = [1, 2, 10, 30, 78, len(xTrain)]
    train_mse = []
    test_mse = []
    ts = []
    for b in bs:
        start = time.time()  # compute the running time
        model = SgdLR(best_lr, b, 1)
        trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
        train_mse.append(trainStats[len(trainStats) - 1]['train-mse'])
        test_mse.append(trainStats[len(trainStats) - 1]['test-mse'])
        end = time.time()
        ts.append(end - start)

    # the closed form solution
    model = StandardLR()
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    train_mse.append(trainStats[0]['train-mse'])
    test_mse.append(trainStats[0]['test-mse'])
    ts.append(trainStats[0]['time'])

    plt.plot(ts, train_mse, '-o', label='train mse')
    plt.plot(ts, test_mse, '-s', label='test mse')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('mse')
    plt.savefig('batch_size.png')

if __name__ == "__main__":
    main()

