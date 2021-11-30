import argparse
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


class RandomForest(object):
    nest = 0           # number of trees
    maxFeat = 0        # maximum number of features
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion

    def __init__(self, nest, maxFeat, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        nest: int
            Number of trees to have in the forest
        maxFeat: int
            Maximum number of features to consider in each tree
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.nest = nest
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample
        self.maxFeat = maxFeat
        self.forest = []

    def train(self, xFeat, y):
        """
        Train the random forest using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the number of trees and
            the values are the out of bag errors
        """
        stats = {}
        n, d = xFeat.shape
        for i in range(self.nest):
            # bootstrap training dataset and choose a random subset of the features
            k_indices = np.random.choice(n, n, replace=True)
            idx = np.random.choice(d, self.maxFeat, replace=True)
            xSub, ySub = xFeat[k_indices, :], y[k_indices]
            xSub = xSub[:, idx]

            # build a tree using decision tree with scikit learn
            tree = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.maxDepth, min_samples_leaf=self.minLeafSample)
            tree = tree.fit(xSub, ySub)

            # add the tree into the forest
            self.forest.append(tree)
            self.forest[i].feature_indices = idx

            # compute the obb error using the out of bag samples
            xOut, yOut = np.delete(xFeat, k_indices, axis=0), np.delete(y, k_indices, axis=0)
            yHat = []

            y_preds = []
            for j in range(i):
                idx = self.forest[j].feature_indices
                sub_X = xOut[:, idx]
                y_pre = self.forest[j].predict(sub_X)
                y_preds.append(y_pre)
            y_preds = np.array(y_preds).T
            
            for y_p in y_preds:
                yHat.append(np.bincount(y_p.astype('int')).argmax())
            print(len(yOut), len(yHat))
            length = len(yOut)
            accuracy = np.sum(yOut == yHat) / length
            obb_error = 1-accuracy
            # mistake = 0
            # for i in range(length):
            #     if yHat[i] != yOut[i]:
            #         mistake += 1
            # obb_error = mistake/length
            
            # obb_error = 1 - np.mean((yOut == yHat))
            stats[i+1] = obb_error

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

        y_preds = []
        for i in range(self.nest):
            idx = self.forest[i].feature_indices
            sub_X = xFeat[:, idx]
            y_pre = self.forest[i].predict(sub_X)
            y_preds.append(y_pre)
        y_preds = np.array(y_preds).T

        for y_p in y_preds:
            yHat.append(np.bincount(y_p.astype('int')).argmax())

        return yHat

def cross_val_error(xTrain, yTrain, clf):
    kf = KFold(n_splits=5, random_state=None)
    data = np.column_stack((xTrain, yTrain))
    errors = []
    n, d = data.shape
    for train, test in kf.split(data):
        clf.train(data[train, 0:d-1], data[train, d-1])
        yHat = clf.predict(data[test, 0:d-1])
        error = 1 - np.mean((data[test, d-1] == yHat))
        errors.append(error)
    return errors


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
    parser.add_argument("xTrain", default='q4xTrain.csv',
                        help="filename for features of the training data")
    parser.add_argument("yTrain", default='q4yTrain.csv',
                        help="filename for labels associated with training data")
    parser.add_argument("xTest", default='q4xTest.csv',
                        help="filename for features of the test data")
    parser.add_argument("yTest", default='q4yTest.csv',
                        help="filename for labels associated with the test data")
    parser.add_argument("--nest", default=5,
                        type=int, help="max number of trees")
    parser.add_argument("--maxFeat", default=5,
                        type=int, help="max number of features")
    parser.add_argument("--criterion", default="gini",
                        type=str, help="split criterion")
    parser.add_argument("--maxDepth", default=4,
                        type=int, help="max depth of trees")
    parser.add_argument("--minLeafSample", default=1,
                        type=int, help="min number of leaf samples")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")
    
    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

# Q2(a)
    np.random.seed(args.seed)   
    model = RandomForest(nest=args.nest, maxFeat=args.maxFeat, criterion=args.criterion, maxDepth=args.maxDepth, minLeafSample=args.minLeafSample)
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)
    yHat = model.predict(xTest)

# Q2(b)
    # Find the best parameters in the wine quality dataset based on the obb error
    print("find the best parameters...")
    # find the best number of trees
    nests = range(1, 50, 1)
    nest_errors = []  # classification errors

    for nest in nests:
        rf = RandomForest(nest=nest, maxFeat=args.maxFeat, criterion=args.criterion,
                         maxDepth=args.maxDepth, minLeafSample=args.minLeafSample)
        errors = cross_val_error(xTrain, yTrain, rf)

        cv_error = np.mean(errors)
        nest_errors.append(cv_error)
    plt.plot(nests, nest_errors)
    plt.xlabel("nest")
    plt.ylabel('error')
    plt.savefig("nest.png")
    plt.show()

    min_error = min(nest_errors)
    best_nest = nests[nest_errors.index(min_error)]
    print("best nest: ", best_nest)

    # find the best max number of features
    maxFeats = range(1, 12)
    maxFeat_errors = []  # classification errors

    for maxFeat in maxFeats:
        rf = RandomForest(nest=best_nest, maxFeat=maxFeat, criterion=args.criterion, maxDepth=args.maxDepth, minLeafSample=args.minLeafSample)
        errors = cross_val_error(xTrain, yTrain, rf)
        cv_error = np.mean(errors)
        maxFeat_errors.append(cv_error)

    plt.plot(maxFeats, maxFeat_errors)
    plt.xlabel("maxFeat")
    plt.ylabel('error')
    plt.savefig("maxFeat.png")
    plt.show()

    min_error = min(maxFeat_errors)
    best_maxFeat = maxFeats[maxFeat_errors.index(min_error)]
    print("best maxFeat: ", best_maxFeat)

    # find the best max depth and min number of leaf samples
    maxDepths = range(1, 12)
    minLeafSamples = range(1, 12)
    errors = []  # classification errors

    for i in range(len(maxDepths)):
        maxDepth = maxDepths[i]
        error = []
        for j in range(len(minLeafSamples)):
            minLeafSample = minLeafSamples[j]
            rf = RandomForest(nest=best_nest, maxFeat=best_maxFeat, criterion=args.criterion, maxDepth=maxDepth, minLeafSample=minLeafSample)
            cv_errors = cross_val_error(xTrain, yTrain, rf)
            cv_error = np.mean(cv_errors)
            error.append(cv_error)
        errors.append(error)
    for i in range(len(maxDepths)):
        plt.plot(minLeafSamples, errors[i], label="maxDepth=" + str(maxDepths[i]))
    plt.legend()
    plt.xlabel("minLeafSamples")
    plt.ylabel('error')
    plt.savefig("Depth_Leaf.png")
    plt.show()

    maxDepthId, minLeafSampleId = np.where(errors == np.min(errors))
    best_maxDepth = maxDepths[maxDepthId[0]]
    best_minLeafSample = minLeafSamples[minLeafSampleId[0]]
    print("best maxDepth: ", best_maxDepth)
    print("best min leaf samples: ", best_minLeafSample)

# Q2(c)
    print("training the model using the optimal parameters")
    bestRF = RandomForest(nest=best_nest, maxFeat=best_maxFeat, criterion=args.criterion, maxDepth=best_maxDepth, minLeafSample=best_minLeafSample)
    bestStats = bestRF.train(xTrain, yTrain)
    print("Best parameters are...")
    print("best nest: ", best_nest)
    print("best maxFeat: ", best_maxFeat)
    print("best maxDepth: ", best_maxDepth)
    print("best min leaf samples: ", best_minLeafSample)
    print(bestStats)
    print("estimated OOB error: ", bestStats[best_nest])
    yHat = bestRF.predict(xTest)
    print("testing error: ", 1 - np.mean(yTest == yHat))

if __name__ == "__main__":
    main()