
import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# load iris dataset
iris = datasets.load_iris()
# create a panda dataframe for iris dataset
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# make target a variable of the dataframe
df['target'] = pd.Series(iris.target)
# create a 'species' variable for different targets
df['species'] = df['target'].replace({0:'Iris Setosa', 1:'Iris Versicolour', 2:'Iris Virginica'})
# create a boxplot for different species' sepal length
sns.boxplot(x=df['species'], y=df['sepal length (cm)'])
plt.show()
# create a boxplot for different species' sepal width
sns.boxplot(x=df['species'], y=df['sepal width (cm)'])
plt.show()
# create a boxplot for different species' petal length
sns.boxplot(x=df['species'], y=df['petal length (cm)'])
plt.show()
# create a boxplot for different species' petal width
sns.boxplot(x=df['species'], y=df['petal width (cm)'])
plt.show()

# create a scatterplot for different species' sepal length and width
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='species')
plt.show()
# create a scatterplot for different species' petal length and width
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='species')
plt.show()

# (d) Iris with short sepal length, short petal length and width can be classified as Setosa.
# Iris with medium sepal length, petal length and width can be classified as Versicolour.
# Iris with long sepal length, petal length and width can be classified as Virginica.
