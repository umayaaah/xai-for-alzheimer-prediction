import pandas as pd
from pyrsistent import immutable
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import pickle 

class DataLoader():
    def __init__(self):
        self.data = None
    
    def load_data(self, path):
        self.data = pd.read_csv(path)
    
    def get_data_split(self, label):
        # Features
        X = self.data.drop([label], axis=1)
        # Label (AD/MCI)
        y = self.data[label]
        return train_test_split(X, y, test_size=0.20, random_state=2021)

    def show_data(self):
        print(self.data.head())

    def oversample_data(self, X_train, y_train):
        oversample = RandomOverSampler(sampling_strategy='minority')
        x_np = X_train.to_numpy()
        y_np = y_train.to_numpy()
        x_np, y_np = oversample.fit_resample(x_np, y_np)
        x_pd = pd.DataFrame(x_np, columns=X_train.columns)
        y_pd = pd.Series(y_np, name=y_train.name)
        return x_pd, y_pd

class RunModel():
    def __init__(self):
        self.model = None
    
    def run_model(self, X_train, X_test, y_train):
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X_train, y_train)
        self.prediction = self.model.predict(X_test)

    def model_performance(self, X_test, y_test):
        self.accuracy = accuracy_score(y_test, self.prediction)
        
        self.f1 = f1_score(y_test, self.prediction)

        self.probs = self.model.predict_proba(X_test)
        self.probs = self.probs[:, 1]
        self.auc = roc_auc_score(y_test, self.probs)
        # print('AUROC = %.3f' % (self.auc))

    def save_model(self, path, X_train, X_test, y_train, y_test):
        pickle.dump([self.model, X_train, X_test, y_train, y_test], open(path, 'wb'))
        print("Saved.")

    def save_obj(self, path, X_train, X_test, y_train, y_test):
        pickle.dump([self, X_train, X_test, y_train, y_test], open(path, 'wb'))
        print("Saved.")