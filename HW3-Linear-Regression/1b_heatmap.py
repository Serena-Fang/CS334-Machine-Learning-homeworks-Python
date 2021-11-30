import seaborn as sb
import scipy
import pandas as pd
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt

xTrain = pd.read_csv('eng_xTrain.csv')
df = xTrain
df['date1'] = pd.to_datetime(df['date'])
df['hour'] = df['date1'].dt.hour
df.loc[df['date1'].dt.dayofweek < 5, 'weekday'] = True
df.loc[df['date1'].dt.dayofweek >= 5, 'weekday'] = False
df = df.drop(columns=['date'])
df = df.drop(columns=['date1'])

y = pd.read_csv('eng_yTrain.csv')
df['target'] = y
plt.figure(figsize=(15,15))
sb.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.savefig('heatmap.jpg')

