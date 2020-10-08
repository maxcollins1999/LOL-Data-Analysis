### Preamble ###################################################################
#
# Author            : Max Collins
#
# Github            : https://github.com/maxcollins1999
#
# Title             : pipeline_utils.py
#
# Date Last Modified: 8/10/2020
#
# Notes             : Contains custom transformer classes for use with the 
#                     sklearn pipeline library.
#
################################################################################

### Imports ####################################################################

#Global
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

import numpy as np
import pandas as pd

################################################################################

class ColumnRemover(BaseEstimator, TransformerMixin):
    """Removes specified columns from a dataset.
    """

    def __init__(self, rm_cols):
        """Takes an integers denoting which columns to remove.
        """

        if type(rm_cols) == list:
            self.rm_cols = rm_cols
        else:
            raise TypeError('rm_cols should be a list')

### Public Methods #############################################################

    def fit(self, X = None, y = None):
        """Present for consistency with sklearn api.
        """

        return self


    def transform(self, X, y = None):
        """Determines columns to remove and removes specified columns from 
        input. 
        """

        return np.delete(np.array(X), self.rm_cols, axis = 1)


    def fit_transform(self, X, y = None):
        """Removes specified columns from input.
        """

        return np.delete(np.array(X), self.rm_cols, axis = 1)

################################################################################

################################################################################

class Clusterer(BaseEstimator, TransformerMixin):
    """Clusters the dataset and returns the dataset with a cluster column.
    """

    def __init__(self, **kwargs):
        """Takes a dictionary of arguments to give to the KMeans clusterer.
        """

        self.kmeans = KMeans(**kwargs)

### Public Methods #############################################################

    def fit(self, X = None, y = None):
        """Takes the X data and fits the clustering model.
        """

        X_array = np.array(X)
        self.kmeans.fit(X_array)
        return self


    def transform(self, X, y = None):
        """Takes X data, appends corresponding clusters and returns the data.
        """

        X_array = np.array(X)
        n = X_array.shape[0]
        return np.append(X_array, self.kmeans.predict(X_array).reshape((n,1)), axis=1)


    def fit_transform(self, X, y = None):
        """Removes specified columns from input.
        """

        X_array = np.array(X)
        n = X_array.shape[0]
        self.kmeans.fit(X_array)
        return np.append(X_array, self.kmeans.predict(X_array).reshape((n,1)), axis=1)

################################################################################

### Tests ######################################################################

if __name__ == "__main__":
    #Test 1: ColumnRemover transform and fit_transform function - array input
    tpass = False
    tdata = np.array([[1,2,3],[4,5,6],[7,8,9]])
    try:
        rm = ColumnRemover(rm_cols=[0,2])
        tmp = rm.fit_transform(tdata)
        if tmp.shape == (3,1):
            if tmp[0,0] == 2 and tmp[1,0] == 5 and tmp[2,0] == 8:
                tmp = rm.transform(tdata)
                if tmp[0,0] == 2 and tmp[1,0] == 5 and tmp[2,0] == 8:
                    tpass = True
    except:
        pass
    if tpass:
        print("Test 1  Passed: ColumnRemover transform and fit_transform function - array input")
    else:
        print("Test 1  Failed: ColumnRemover transform and fit_transform function - array input")

    #Test 2: ColumnRemover transform and fit_transform function - pandas input
    tpass = False
    tdata = pd.DataFrame(np.array([[1,2,3],[4,5,6],[7,8,9]]))
    try:
        rm = ColumnRemover(rm_cols=[0,2])
        tmp = rm.fit_transform(tdata)
        if tmp.shape == (3,1):
            if tmp[0,0] == 2 and tmp[1,0] == 5 and tmp[2,0] == 8:
                tmp = rm.transform(tdata)
                if tmp[0,0] == 2 and tmp[1,0] == 5 and tmp[2,0] == 8:
                    tpass = True
    except:
        pass
    if tpass:
        print("Test 2  Passed: ColumnRemover transform and fit_transform function - pandas input")
    else:
        print("Test 2  Failed: ColumnRemover transform and fit_transform function - pandas input")

    #Test 3: Clusterer transform and fit_transform function - array input
    tpass = False
    tdata = np.array([[1,2],[1,3],[5,7]])
    try:
        clst = Clusterer(**{'n_clusters' : 2, 'random_state' : 783})
        tr_tmp = np.append(tdata, np.array([[0],[0],[1]]),axis=1)
        tmp = clst.fit_transform(tdata)
        if np.array_equal(tmp, tr_tmp):
            clst.fit(tdata)
            tmp = clst.transform(tdata)
            if np.array_equal(tmp, tr_tmp):
                tpass = True
    except:
        pass
    if tpass:
        print("Test 3  Passed: Clusterer transform and fit_transform function - array input")
    else:
        print("Test 3  Failed: Clusterer transform and fit_transform function - array input")

    #Test 4: Clusterer transform and fit_transform function - pandas input
    tpass = False
    tdata = pd.DataFrame(np.array([[1,2],[1,3],[5,7]]))
    try:
        clst = Clusterer(**{'n_clusters' : 2, 'random_state' : 783})
        tr_tmp = np.append(tdata, np.array([[0],[0],[1]]),axis=1)
        tmp = clst.fit_transform(tdata)
        if np.array_equal(tmp, tr_tmp):
            clst.fit(tdata)
            tmp = clst.transform(tdata)
            if np.array_equal(tmp, tr_tmp):
                tpass = True
    except:
        pass
    if tpass:
        print("Test 4  Passed: Clusterer transform and fit_transform function - pandas input")
    else:
        print("Test 4  Failed: Clusterer transform and fit_transform function - pandas input")

################################################################################