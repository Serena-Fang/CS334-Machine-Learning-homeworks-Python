import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
import time
import time

 
def holdout(model, xFeat, y, testSize):
    trainAuc = 0
    testAuc = 0
    timeElapsed = 0
    # TODO fill int
    start = time.time()
    xTrain, xTest, yTrain, yTest = train_test_split(xFeat, y, test_size=testSize)
    model = model.fit(xTrain, yTrain)
    trainAuc = metrics.roc_auc_score(yTrain, model.predict(xTrain))
    testAuc = metrics.roc_auc_score(yTest, model.predict(xTest))

    timeElapsed = time.time() - start
    return trainAuc, testAuc, timeElapsed


def kfold_cv(model, xFeat, y, k):
    trainAuc = 0
    testAuc = 0
    timeElapsed = 0
    # TODO FILL IN
    start = time.time()
    kf = KFold(n_splits = k, random_state=2020)
    trainAucs = []
    testAucs = []
    xFeat = xFeat.values
    y = y.values
    for train, test in kf.split(xFeat):
        xTrain, xTest, yTrain, yTest = xFeat[train], xFeat[test], y[train], y[test]
        model = model.fit(xTrain, yTrain)
        trainAucs.append(metrics.roc_auc_score(yTrain, model.predict(xTrain)))
        testAucs.append(metrics.roc_auc_score(yTest, model.predict(xTest)))

    trainAuc = np.mean(trainAucs)
    testAuc = np.mean(testAucs)
    timeElapsed = time.time() - start
    return trainAuc, testAuc, timeElapsed


def mc_cv(model, xFeat, y, testSize, r):
    trainAuc = 0
    testAuc = 0
    timeElapsed = 0
    # TODO FILL IN
    start = time.time()
    for i in range(r):
        trainauc, testauc, _ = holdout(model, xFeat, y, testSize)
        trainAuc += trainauc
        testAuc += testauc
    trainAuc = trainAuc/r
    testAuc = testAuc/r
    timeElapsed = time.time() - start
    return trainAuc, testAuc, timeElapsed


def sktree_train_test(model, xTrain, yTrain, xTest, yTest):
    """
    Given a sklearn tree model, train the model using
    the training dataset, and evaluate the model on the
    test dataset.

    Parameters
    ----------
    model : DecisionTreeClassifier object
        An instance of the decision tree classifier 
    xTrain : nd-array with shape nxd
        Training data
    yTrain : 1d array with shape n
        Array of labels associated with training data
    xTest : nd-array with shape mxd
        Test data
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    trainAUC : float
        The AUC of the model evaluated on the training data.
    testAuc : float
        The AUC of the model evaluated on the test data.
    """
    # fit the data to the training dataset
    model.fit(xTrain, yTrain)
    # predict training and testing probabilties
    yHatTrain = model.predict_proba(xTrain)
    yHatTest = model.predict_proba(xTest)
    # calculate auc for training
    fpr, tpr, thresholds = metrics.roc_curve(yTrain['label'],
                                             yHatTrain[:, 1])
    trainAuc = metrics.auc(fpr, tpr)
    # calculate auc for test dataset
    fpr, tpr, thresholds = metrics.roc_curve(yTest['label'],
                                             yHatTest[:, 1])
    testAuc = metrics.auc(fpr, tpr)
    return trainAuc, testAuc

def knnKfold(k_range, xFeat, y):
    bestK = k_range[0]
    bestScore = 0
    xFeat = xFeat.values
    y = y.values
    kf = KFold(n_splits=10, random_state=2020)

    for k in k_range:
        Score = 0

        for train, test in kf.split(xFeat):
            knn = KNeighborsClassifier(n_neighbors=k)
            xTrain, xTest, yTrain, yTest = xFeat[train], xFeat[test], y[train], y[test]
            knn = knn.fit(xTrain, yTrain)
            Score += knn.score(xTest, yTest)
        avgScore = Score/5

        if avgScore > bestScore:
            bestK = k
            bestScore = avgScore
    return bestK, bestScore

def treeKfold(mds, mls, xFeat, y):
    best_md = mds[0]
    best_mls = mls[0]
    bestScore = 0
    xFeat = xFeat.values
    y = y.values
    kf = KFold(n_splits = 10, random_state=2020)

    for md in mds:
        for ml in mls:
            Score = 0

            for train, test in kf.split(xFeat):
                tree = DecisionTreeClassifier(max_depth=md,
                                              min_samples_leaf=ml)
                xTrain, xTest, yTrain, yTest = xFeat[train], xFeat[test], y[train], y[test]
                tree = tree.fit(xTrain, yTrain)
                Score += tree.score(xTest, yTest)
            avgScore = Score/5

            if avgScore > bestScore:
                best_md = md
                best_mls = ml
                bestScore = avgScore

    return best_md, best_mls, bestScore

def bestKnn(bestK, xTrain, yTrain, xTest, yTest, removeRate):
    if removeRate > 0:
        xTrain, _, yTrain, _ = train_test_split(xTrain, yTrain, test_size = removeRate)

    knn = KNeighborsClassifier(n_neighbors=bestK)
    knn = knn.fit(xTrain, yTrain)

    knnAuc = metrics.roc_auc_score(yTest, knn.predict(xTest))
    knnAcc = metrics.accuracy_score(yTest, knn.predict(xTest))
    print("k-nn randomly remove: %.2f" % removeRate, "testing AUC: %.4f" % knnAuc,
          "testing accuracy: %.4f" % knnAcc)
    return knnAuc, knnAcc

def bestTree(best_md, best_mls, xTrain, yTrain, xTest, yTest, removeRate):
    if removeRate > 0:
        xTrain, _, yTrain, _ = train_test_split(xTrain, yTrain, test_size = removeRate)

    tree = DecisionTreeClassifier(max_depth = best_md, min_samples_leaf = best_mls)
    tree = tree.fit(xTrain, yTrain)

    treeAuc = metrics.roc_auc_score(yTest, tree.predict(xTest))
    treeAcc = metrics.accuracy_score(yTest, tree.predict(xTest))
    print("Decision Tree randomly remove: %.2f" % removeRate, "testing AUC: %.4f" % treeAuc,
          "testing accuracy: %.4f" % treeAcc)
    return treeAuc, treeAcc


def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
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
    # create the decision tree classifier
    dtClass = DecisionTreeClassifier(max_depth=15,
                                     min_samples_leaf=10)
    # use the holdout set with a validation size of 30 of training
    aucTrain1, aucVal1, time1 = holdout(dtClass, xTrain, yTrain, 0.70)
    # use 2-fold validation
    aucTrain2, aucVal2, time2 = kfold_cv(dtClass, xTrain, yTrain, 2)
    # use 5-fold validation
    aucTrain3, aucVal3, time3 = kfold_cv(dtClass, xTrain, yTrain, 5)
    # use 10-fold validation
    aucTrain4, aucVal4, time4 = kfold_cv(dtClass, xTrain, yTrain, 10)
    # use MCCV with 5 samples
    aucTrain5, aucVal5, time5 = mc_cv(dtClass, xTrain, yTrain, 0.70, 5)
    # use MCCV with 10 samples
    aucTrain6, aucVal6, time6 = mc_cv(dtClass, xTrain, yTrain, 0.70, 10)
    # train it using all the data and assess the true value
    trainAuc, testAuc = sktree_train_test(dtClass, xTrain, yTrain, xTest, yTest)
    perfDF = pd.DataFrame([['Holdout', aucTrain1, aucVal1, time1],
                           ['2-fold', aucTrain2, aucVal2, time2],
                           ['5-fold', aucTrain3, aucVal3, time3],
                           ['10-fold', aucTrain4, aucVal4, time4],
                           ['MCCV w/ 5', aucTrain5, aucVal5, time5],
                           ['MCCV w/ 10', aucTrain6, aucVal6, time6],
                           ['True Test', trainAuc, testAuc, 0]],
                           columns=['Strategy', 'TrainAUC', 'ValAUC', 'Time'])
    print(perfDF)

    # q3 (a)
    k_range = range(1,26)
    bestK, bestScore = knnKfold(k_range, xTrain, yTrain)
    print("Find the optimal hyperparameters for knn")
    print("current best score: %.2f" % bestScore + ", best k: %d" % bestK)

    mds = [2,4,6,8,10]
    mls = [1,3,5,7,9]
    best_md, best_mls, bestScore = treeKfold(mds, mls, xTrain, yTrain)
    print("Find the optimal hyperparameters for decision tree")
    print("current best score is: %.2f" % bestScore, "best max depth: %d" % best_md,
          "best minimum samples of leaves: %d" % best_mls)

    # q3(b)
    knnAuc = [0,0,0,0]
    knnAcc = [0,0,0,0]
    removeRates = [0, 0.05, 0.1, 0.2]
    for removeRate in removeRates:
        i = removeRates.index(removeRate)
        knnAuc[i], knnAcc[i] = bestKnn(bestK, xTrain, yTrain, xTest, yTest, removeRate)

    # q3(c)
    treeAuc = [0,0,0,0]
    treeAcc = [0,0,0,0]
    for removeRate in removeRates:
        i = removeRates.index(removeRate)
        treeAuc[i], treeAcc[i] = bestTree(best_md, best_mls, xTrain, yTrain, xTest, yTest, removeRate)

    # q3(d)
    result = pd.DataFrame([['k-nn', 0, knnAuc[0], knnAcc[0]],
                       ['k-nn', 0.05, knnAuc[1], knnAcc[1]],
                       ['k-nn', 0.1, knnAuc[2], knnAcc[2]],
                       ['k-nn', 0.2, knnAuc[3], knnAcc[3]],
                       ['decision tree', 0, treeAuc[0], treeAcc[0]],
                       ['decision tree', 0.05, treeAuc[1], treeAcc[1]],
                       ['decision tree', 0.1, treeAuc[2], treeAcc[2]],
                       ['decision tree', 0.2, treeAuc[3], treeAcc[3]]],
                       columns=['model', 'removeRate', 'TestAuc', 'TestAcc'])
    print(result)

if __name__ == "__main__":
    main()
