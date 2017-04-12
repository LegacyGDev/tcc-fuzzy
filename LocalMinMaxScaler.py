import numpy as np

class MinMaxScaler():
    def fit(self, X, M):
        data_min = np.min(X)
        data_max = np.max(X)
        data_range = data_max - data_min
        self.data_min = data_min - (data_range * (M/100))
        self.data_max = data_max + (data_range * (M/100))
        return self

    def transform(self, X):
        X_transformed = []
        for item in X:
            X_transformed.append( (item - self.data_min)/(self.data_max - self.data_min) )
        return np.asarray(X_transformed)


    def inverse_transform(self, X):
        X_reversed = []
        for item in X:
            X_reversed.append( item*(self.data_max - self.data_min) + self.data_min )
        return np.asarray(X_reversed)
