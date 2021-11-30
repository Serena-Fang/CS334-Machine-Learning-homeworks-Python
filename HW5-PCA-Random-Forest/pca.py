import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import heapq

def normalize(xTrain, xTest):
    scaler = StandardScaler().fit(xTrain)
    xTrain = scaler.transform(xTrain)
    xTest = scaler.transform(xTest)
    return xTrain, xTest

def train_predict(xTrain, yTrain, xTest):
    lg = LogisticRegression()
    lg.fit(xTrain, yTrain)
    yPred = lg.predict_proba(xTest)
    yPred = yPred[:, 1]
    return yPred

def file_to_numpy(filename):
    df = pd.read_csv(filename)
    return df.to_numpy()

xTrain = file_to_numpy('q4xTrain.csv')
xTest = file_to_numpy('q4xTest.csv')
yTrain = file_to_numpy('q4yTrain.csv')
yTest = file_to_numpy('q4yTest.csv')

# Q1(a)
xTrain_n, xTest_n = normalize(xTrain, xTest)
yPred_n = train_predict(xTrain_n, yTrain, xTest_n)

# Q1(b)
pca = decomposition.PCA(n_components=10)
pca.fit_transform(xTrain_n)
print("The probabilities are: ")
print(pca.explained_variance_ratio_.cumsum())

xTrain = pd.read_csv('q4xTrain.csv')
features = []
for col in xTrain.columns:
    features.append(col)
df_components = pd.DataFrame(abs(pca.components_))
df_components = df_components.set_axis(features, axis=1, inplace=False)
df_components.to_csv('features.csv')
df_list = df_components.values.tolist()
com = 1
for n in df_list:
    print('Three most important features and probability in ' + str(com) + ' principal component are:')
    print(heapq.nlargest(3, zip(n, features)))
    com += 1
    if com > 3:
        break

# Q1(c)
xTrain_pca = pca.fit_transform(xTrain_n)
xTest_pca = pca.fit_transform(xTest_n)
yPred_pca = train_predict(xTrain_pca, yTrain, xTest_pca)

fpr_n, tpr_n, _ = roc_curve(yTest, yPred_n)
fpr_pca, tpr_pca, _ = roc_curve(yTest, yPred_pca)
plt.plot(fpr_n,tpr_n,label="normal")
plt.plot(fpr_pca,tpr_pca,label="pca")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve on normal and pca data')
plt.legend(loc="lower right")
plt.savefig("roc.png")
plt.show()