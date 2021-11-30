import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def extract_features(df):
    df['date1'] = pd.to_datetime(df['date'])
    df['hour'] = df['date1'].dt.hour
    df = df.drop(columns=['date'])
    df = df.drop(columns=['date1'])
    return df

def select_features(df):
    df.drop(['Visibility', 'T6', 'RH_4'], axis=1, inplace=True)
    return df


def preprocess_data(trainDF, testDF):
    sc = StandardScaler()
    trainDF[trainDF.columns] = sc.fit_transform(trainDF)
    testDF[testDF.columns] = sc.fit_transform(testDF)
    return trainDF, testDF


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("outTrain",
                        help="filename of the updated training data")
    parser.add_argument("outTest",
                        help="filename of the updated test data")
    parser.add_argument("--trainFile",
                        default="eng_xTrain.csv",
                        help="filename of the training data")
    parser.add_argument("--testFile",
                        default="eng_xTest.csv",
                        help="filename of the test data")
    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.trainFile)
    xTest = pd.read_csv(args.testFile)
    # extract the new features
    xNewTrain = extract_features(xTrain)
    xNewTest = extract_features(xTest)
    # select the features
    xNewTrain = select_features(xNewTrain)
    xNewTest = select_features(xNewTest)
    # preprocess the data
    xTrainTr, xTestTr = preprocess_data(xNewTrain, xNewTest)
    # save it to csv
    xTrainTr.to_csv(args.outTrain, index=False)
    xTestTr.to_csv(args.outTest, index=False)


if __name__ == "__main__":
    main()
