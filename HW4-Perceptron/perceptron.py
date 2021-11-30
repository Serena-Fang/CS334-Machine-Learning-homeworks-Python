import argparse
import numpy as np
import pandas as pd
import time

class Perceptron(object):
    mEpoch = 1000  # maximum epoch size
    w = None       # weights of the perceptron

    def __init__(self, epoch):
        self.mEpoch = epoch

    def train(self, xFeat, y):
        """
        Train the perceptron using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the epochs and values the number of mistakes
        """
        stats = {}
        # TODO implement this
        self.w = np.zeros(1+xFeat.shape[1])

        for epoch in range(self.mEpoch):
            mistakes = 0
            for i in range(len(xFeat)):
                delta_w = 0.01 * (y[i] - self.predict(xFeat[i]))
                self.w[1:] += delta_w * xFeat[i]
                self.w[0] += delta_w
                mistakes += int(delta_w != 0.0)
            if mistakes == 0:
                break
            stats[epoch] = mistakes
        return stats

    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted response per sample
        """
        yHat = []
        yHat = np.dot(xFeat, self.w[1:]) + self.w[0]
        yHat = np.where(yHat >= 0.0, 1, 0)
        return yHat

    def pos_neg_words(self, data):
        weights = self.w
        words = list(data.columns.values)
        np_words = np.array(words)
        
        positive_index = np.argsort(-weights)[:15]
        pos_list = np_words[positive_index]
        negative_index = np.argsort(weights)[:15]
        neg_list = np_words[negative_index]
        return pos_list, neg_list

def calc_mistakes(yHat, yTrue):
    """
    Calculate the number of mistakes
    that the algorithm makes based on the prediction.

    Parameters
    ----------
    yHat : 1-d array or list with shape n
        The predicted label.
    yTrue : 1-d array or list with shape n
        The true label.      

    Returns
    -------
    err : int
        The number of mistakes that are made
    """
    mistake = 0
    for i in range(len(yTrue)):
        if yHat[i] != yTrue[i]:
            mistake += 1
    return mistake


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain", default="binary_train.csv",
                        help="filename for features of the training data")
    parser.add_argument("yTrain", default="y_train.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest", default="binary_test.csv",
                        help="filename for features of the test data")
    parser.add_argument("yTest", default="y_test.csv",
                        help="filename for labels associated with the test data")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    np.random.seed(args.seed)   
    model = Perceptron(args.epoch)
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)
    yHat = model.predict(xTest)
    # print out the number of mistakes
    print("Number of mistakes on the test dataset")
    print(calc_mistakes(yHat, yTest))
    
    df = pd.read_csv(args.xTrain)
    pos_list, neg_list = model.pos_neg_words(df)
    print("15 words with most positive weights: ")
    print(pos_list)

    print("15 words with most negative weights: ")
    print(neg_list)


if __name__ == "__main__":
    main()