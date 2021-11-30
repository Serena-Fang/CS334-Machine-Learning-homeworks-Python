from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from perceptron import Perceptron

def calc_mistakes(yHat, yTrue):
    mistake = 0
    for i in range(len(yTrue)):
        if yHat[i] != yTrue[i]:
            mistake += 1
    return mistake

def kfold_epoch(x, y, mEpoch_list):
    epoch_mistake = {}
    k = len(mEpoch_list)
    kf = KFold(n_splits=k)
    for mEpoch in mEpoch_list:
        mistakes_list = []
        for train_i, test_i in kf.split(x):
            xTrain, xTest = x[train_i], x[test_i]
            yTrain, yTest = y[train_i], y[test_i]
            p = Perceptron(mEpoch)
            p.train(xTrain, yTrain)
            yHat = p.predict(xTest)
            mistakes = calc_mistakes(yHat, yTest)
            mistakes_list.append(mistakes)
        avg_mistake = np.mean(mistakes_list)
        epoch_mistake[mEpoch] = avg_mistake
    return epoch_mistake

def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()

def pos_neg_words(data):
    weights = self.w
    words = list(data.columns.values)
    np_words = np.array(words)
    positive_index = np.argsort(-weights)[:15]
    pos_list = np_words[positive_index]
    negative_index = np.argsort(weights)[:15]
    neg_list = np_words[negative_index]
    return pos_list, neg_list

b_train = file_to_numpy('binary_train.csv')
b_test = file_to_numpy('binary_test.csv')
c_train = file_to_numpy('count_train.csv')
c_test = file_to_numpy('count_test.csv')
y_train = file_to_numpy('y_train.csv')
y_test = file_to_numpy('y_test.csv')

mEpoch_list = [5, 10, 15, 20, 30, 50, 100]

b_mistake_list = kfold_epoch(b_train, y_train, mEpoch_list)
c_mistake_list = kfold_epoch(c_train, y_train, mEpoch_list)
print("Binary dataset: Average mistakes on different max epochs")
print(b_mistake_list)
print("Count dataset: Average mistakes on different max epochs")
print(c_mistake_list)

# 20 is the optimal number of epochs for binary dataset
b_model = Perceptron(20)
b_model.train(b_train, y_train)
yHat = b_model.predict(b_test)
# print out the number of mistakes
print("Number of mistakes on the binary test dataset:")
print(calc_mistakes(yHat, y_test))

# 100 is the optimal number of epochs for binary dataset
c_model = Perceptron(100)
c_model.train(c_train, y_train)
yHat = c_model.predict(c_test)
# print out the number of mistakes
print("Number of mistakes on the count test dataset:")
print(calc_mistakes(yHat, y_test))

b_df = pd.read_csv('binary_train.csv')
pos_list, neg_list = b_model.pos_neg_words(b_df)
print("Binary dataset: 15 words with most positive weights: ")
print(pos_list)
print("Binary dataset: 15 words with most negative weights: ")
print(neg_list)

c_df = pd.read_csv('count_train.csv')
pos_list, neg_list = c_model.pos_neg_words(c_df)
print("Count dataset: 15 words with most positive weights: ")
print(pos_list)
print("Count dataset: 15 words with most negative weights: ")
print(neg_list)