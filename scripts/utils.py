import pandas as pd
from pyrsistent import immutable
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

class DataLoader():
    def __init__(self):
        self.data = None
    
    def load_data(self, path):
        self.data = pd.read_csv(path)
    
    def get_data_split(self):
        # Features
        X = self.data.drop(['AD'], axis=1)
        # Label (AD)
        y = self.data['AD']
        return train_test_split(X, y, test_size=0.20, random_state=2021)

    # def data(self):
    #     return self.data

    def oversample_data(self, X_train, y_train):
        oversample = RandomOverSampler(sampling_strategy='minority')
        x_np = X_train.to_numpy()
        y_np = y_train.to_numpy()
        x_np, y_np = oversample.fit_resample(x_np, y_np)
        x_pd = pd.DataFrame(x_np, columns=X_train.columns)
        y_pd = pd.Series(y_np, name=y_train.name)
        return x_pd, y_pd