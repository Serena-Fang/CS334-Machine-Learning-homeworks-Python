import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

mds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mds_train_scores = [0.870420017873101, 0.870420017873101, 0.8731009830205541, 0.8731009830205541, 0.8731009830205541, 0.8731009830205541, 0.8731009830205541, 0.8731009830205541, 0.8731009830205541, 0.8739946380697051]
mds_test_scores = [0.8625, 0.8625, 0.8645833333333334, 0.8645833333333334, 0.8645833333333334, 0.8645833333333334, 0.8645833333333334, 0.8645833333333334, 0.8645833333333334, 0.8645833333333334]
mls_train_scores = [0.8748882931188561, 0.8748882931188561, 0.8739946380697051, 0.8739946380697051, 0.8731009830205541, 0.8731009830205541, 0.8731009830205541, 0.8731009830205541, 0.8731009830205541, 0.8731009830205541]
mls_test_scores = [0.8645833333333334, 0.8645833333333334, 0.8645833333333334, 0.8645833333333334, 0.8645833333333334, 0.8645833333333334, 0.8645833333333334, 0.8645833333333334, 0.8645833333333334, 0.8645833333333334]

plt.title('Training and Test Accuracy for Different Max Depths ')
plt.plot(mds_train_scores, color='blue', label='Train accuracy')
plt.plot(mds_test_scores, color='red', label='Test accuracy')
plt.legend()
plt.xlabel('Max depth')
plt.ylabel('Accuracy')
plt.savefig('md.jpg', dpi=200, bbox_inches='tight')
plt.cla()

plt.title('Training and Test Accuracy for Different Minimum Leaves Samples ')
plt.plot(mls_train_scores, color='blue', label='Train accuracy')
plt.plot(mls_test_scores, color='red', label='Test accuracy')
plt.legend()
plt.xlabel('Minimum Leaves Samples')
plt.ylabel('Accuracy')
plt.savefig('mls.jpg', dpi=200, bbox_inches='tight')