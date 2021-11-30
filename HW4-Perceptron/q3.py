import argparse
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression

def train(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    yHat = model.predict(x_test)
    # print out the number of mistakes
    print("Number of mistakes on the test dataset")
    print(calc_mistakes(yHat, y_test))
    return None

def calc_mistakes(yHat, yTrue):
    mistake = 0
    for i in range(len(yHat)):
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
    b_train = file_to_numpy('binary_train.csv')
    b_test = file_to_numpy('binary_test.csv')
    c_train = file_to_numpy('count_train.csv')
    c_test = file_to_numpy('count_test.csv')
    y_train = file_to_numpy('y_train.csv')
    y_test = file_to_numpy('y_test.csv')
    
    multiNB = MultinomialNB()
    bernoulliNB = BernoulliNB()
    lgr = LogisticRegression()

    print("BernoulliNB on Binary Dataset: ")
    train(bernoulliNB, b_train, b_test, y_train, y_test)
    print("BernoulliNB on Count Dataset: ")
    train(bernoulliNB, c_train, c_test, y_train, y_test)
    print("MultinomialNB on Binary Dataset:")
    train(multiNB, b_train, b_test, y_train, y_test)
    print("MultinomialNB on Count Dataset:")
    train(multiNB, c_train, c_test, y_train, y_test)
    print("Logistic Regression on Binary Dataset:")
    train(lgr, b_train, b_test, y_train, y_test)
    print("Logistic Regression on Count Dataset:")
    train(lgr, c_train, c_test, y_train, y_test)

if __name__ == "__main__":
    main()
