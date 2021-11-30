import numpy as np
import pandas as pd
import argparse
import knn
import matplotlib.pyplot as plt
k = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
train_scores = [1.0, 1.0, 0.945, 0.959, 0.942, 0.955, 0.939, 0.95, 0.932, 0.941, 0.926, 0.94, 0.927, 0.935, 0.92, 0.93, 0.922, 0.931, 0.927, 0.931, 0.929, 0.93, 0.926, 0.927, 0.926]
test_scores = [0.903, 0.903, 0.916, 0.917, 0.925, 0.924, 0.921, 0.916, 0.916, 0.919, 0.918, 0.918, 0.924, 0.919, 0.923, 0.927, 0.926, 0.927, 0.923, 0.928, 0.921, 0.923, 0.922, 0.927, 0.924]

plt.plot(k, train_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Train Accuracy')
plt.show()

plt.plot(k, test_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Test Accuracy')
plt.show()