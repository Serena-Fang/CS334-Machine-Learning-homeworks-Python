import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.metrics import mean_squared_error


class LinearRegression(ABC):
    """
    Base Linear Regression class from which all 
    linear regression algorithm implementations are
    subclasses. Can not be instantiated.
    """
    beta = None      # Coefficients

    @abstractmethod
    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        Train the linear regression and predict the values

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : dictionary
            key refers to the batch number
            value is another dictionary with time elapsed and mse
        """
        pass

    def predict(self, xFeat):
        yHat = []
        # TODO
        n = np.shape(xFeat)[0]
        xFeat = np.concatenate((np.ones((n,1)), xFeat), axis=1)
        yHat = np.matmul(xFeat, self.beta)
        return yHat

    def mse(self, xFeat, y):
        """
        """
        yHat = self.predict(xFeat)
        return mean_squared_error(y, yHat)


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()
