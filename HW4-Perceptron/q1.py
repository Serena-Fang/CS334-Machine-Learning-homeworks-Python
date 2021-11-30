import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


def model_assessment(filename):
    """
    Given the entire data, split it into training and test set 
    so you can assess your different models 
    to compare perceptron, logistic regression,
    and naive bayes. 
    """
    label = []
    email = []
    with open(filename) as file:
        lines = file.readlines()
    for line in lines:
        label.append(line[:1])
        email.append(line[2:-1])
    x_train, x_test, y_train, y_test = train_test_split(email, label, test_size=0.3, random_state=1)
    return x_train, x_test, y_train, y_test


def build_vocab_map(x_train):
    vocab_map = {}
    for email in x_train:
        words = set(email.split())
        for word in words:
            if word in vocab_map.keys():
                vocab_map[word] += 1
            else:
                vocab_map[word] = 1
    sorted_keys = sorted(vocab_map.keys())
    vocab_map = {key:vocab_map[key] for key in sorted_keys}
    freq_vocab = []
    for word, freq in vocab_map.items():
        if freq >= 30:
            freq_vocab.append(word)
    return vocab_map, freq_vocab

def construct_binary(x_train, freq_vocab):
    """
    Construct email datasets based on
    the binary representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is 1 if the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """
    cv = CountVectorizer()
    x_traincv = cv.fit_transform(x_train)
    count = x_traincv.toarray()
    vocab = cv.get_feature_names_out()
    df1 = pd.DataFrame(count, columns=vocab)
    df_freq = df1[df1.columns.intersection(freq_vocab)]
    df_binary = df_freq
    df_binary[df_binary > 0] = 1
    return df_binary

def construct_count(x_train, freq_vocab):
    """
    Construct email datasets based on
    the count representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is the number of times the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """
    cv = CountVectorizer()
    x_traincv = cv.fit_transform(x_train)
    count = x_traincv.toarray()
    vocab = cv.get_feature_names_out()
    df1 = pd.DataFrame(count, columns=vocab)
    df_count = df1[df1.columns.intersection(freq_vocab)]
    return df_count


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="spamAssassin.data",
                        help="filename of the input data")
    args = parser.parse_args()
    x_train, x_test, y_train, y_test = model_assessment(args.data)
    vocab_map, freq_vocab = build_vocab_map(x_train)
    
    binary_train = construct_binary(x_train, freq_vocab)
    binary_train.to_csv('binary_train.csv', index=False)
    
    binary_test = construct_binary(x_test, freq_vocab)
    binary_test.to_csv('binary_test.csv', index=False)
    
    count_train = construct_count(x_train, freq_vocab)
    count_train.to_csv('count_train.csv', index=False)
    
    count_test = construct_count(x_test, freq_vocab)
    count_test.to_csv('count_test.csv', index=False)

    df_yTrain = pd.DataFrame(y_train)
    df_yTest = pd.DataFrame(y_test)
    df_yTrain.to_csv('y_train.csv', index=False)
    df_yTest.to_csv('y_test.csv', index=False)

if __name__ == "__main__":
    main()
