import numpy as np
from scipy.spatial.distance import cdist, squareform
from collections import Counter

class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
#         В зависимости от выбранного способа рассчитываем массив расстояние
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)
        # Если массив образов соответствия 0 представлен булевыми значеними
#         то получаем булевый массив отклонения от 0
        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        '''
        Computes distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                dists[i_test][i_train]=np.sum(np.abs(self.train_X[i_train] - X[i_test]))
        return dists

    def compute_distances_one_loop(self, X):
        '''
        Computes distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            dists[i_test]=np.abs(self.train_X - X[i_test]).sum(axis=1)
        return dists

    def compute_distances_no_loops(self, X):
        '''
        Computes distance from every sample of X to every training sample
        Fully vectorizes the calculations

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        # Using float32 to to save memory - the default is float64
        dists = np.zeros((num_test, num_train), np.float32)
        dists = (np.abs(X[:,np.newaxis,:] - self.train_X[np.newaxis, :])).sum(axis=2)
        return dists

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        for i in range(num_test):
#             print(i)
#             получаем индексы минимальных self.k значений
            idx = np.argpartition(dists[i], self.k)
            #  Получаем минимальные первые К индексов 
            min_k = idx[:self.k]
#             display(min_k)
            # Получаем булевый ответ 0 или нет соотв. индекс
            all_value = self.train_y[min_k]
#             display(all_value)
            count_y_0 = (all_value==True).sum()
#             display(count_y_0) 
            pred[i] = count_y_0 >= self.k - count_y_0
        return pred

    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        num_test = dists.shape[0]
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)
        for i in range(num_test):
            # Implement choosing best class based on k
            # nearest training samples
            idx = np.argpartition(dists[i], self.k)
            min_k = idx[:self.k]
            all_value = self.train_y[min_k]
            value_cat = Counter(all_value)
#             print(value_cat.most_common(1)[0][0])
            pred[i]=value_cat.most_common(1)[0][0]
        return pred
