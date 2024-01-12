# write your code here
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
import math


class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.coef_ = None
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    def sigmoid(self, t):
        return 1.0 / (1.0 + np.exp(-t))

    def predict_proba(self, row, coef_):
        t = np.dot(row, coef_[1:])
        t += coef_[0] if self.fit_intercept else 0
        return self.sigmoid(t)

    def fit_mse(self, x_train, y_train):
        self.coef_ = np.zeros(x_train.shape[1] + 1)  # initialized weights
        n = x_train.shape[0]
        for ep in range(self.n_epoch):
            for y, x in zip(y_train, x_train):
                # update all weights
                y_hat = self.predict_proba(x, self.coef_)
                k = self.l_rate * (y_hat - y) * y_hat * (1 - y_hat)
                self.coef_[0] -= k
                self.coef_[1:] -= k * x
            if ep == 0:
                error_first = [ (y - self.predict_proba(x, self.coef_))**2/n for y, x in zip(y_train, x_train) ]
        error_last = [(y - self.predict_proba(x, self.coef_))**2/n for y, x in zip(y_train, x_train)]
        return {'mse_error_first': error_first,'mse_error_last': error_last}
    def fit_log_loss(self, x_train, y_train):
        n = x_train.shape[0]
        self.coef_ = np.zeros(x_train.shape[1] + 1)  # initialized weights
        for ep in range(self.n_epoch):
            for y, x in zip(y_train, x_train):
                # update all weights
                y_hat = self.predict_proba(x, self.coef_)
                k = self.l_rate * (y_hat - y) / n
                self.coef_[0] -= k
                self.coef_[1:] -= k * x
            if ep == 0:
                y_hat = np.array([self.predict_proba(x, self.coef_) for x in x_train])
                error_first = -1 * (y_train * np.log(y_hat) + (1 - y_train) * np.log(1 - y_hat)) / n
        y_hat = np.array([self.predict_proba(x, self.coef_) for y, x in zip(y_train, x_train)])
        error_last = -1 * (y_train * np.log(y_hat) + (1 - y_train) * np.log(1 - y_hat)) / n
        return {'logloss_error_first': list(error_first), 'logloss_error_last': list(error_last)}

    def predict(self, x, cut_off=0.5) -> list[int]:
        # predictions are binary values - 0 or 1
        return np.array([0 if self.predict_proba(row, self.coef_) < cut_off else 1 for row in x])


def prepare_data(selected_features: list, train_size: float = 0.8, random_state: int = 43):
    # Load the Breast Cancer Wisconsin dataset
    data = load_breast_cancer()
    # Extract the features and target variable
    x = data.data  # Features
    y = data.target  # Target variable (0: malignant, 1: benign)
    # Select only the "worst concave points" and "worst perimeter" features
    df = pd.DataFrame(x, columns=data.feature_names)
    x_selected = df[selected_features]
    # Standardize the features
    scaler = StandardScaler()
    x_standardized = scaler.fit_transform(x_selected)
    # Split the dataset into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x_standardized, y, train_size=train_size, random_state=random_state
    )
    return x_train, x_test, y_train, y_test


def stage1():
    x_train, x_test, y_train, y_test = prepare_data(["worst concave points", "worst perimeter"])
    clr = CustomLogisticRegression()
    res = [clr.predict_proba(x, np.array([0.77001597, -2.12842434, -2.39305793])) for x in x_test[:10]]
    print(res)


def stage2():
    x_train, x_test, y_train, y_test = prepare_data(["worst concave points", "worst perimeter", "worst radius"])
    lr = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)
    lr.fit_mse(x_train, y_train)
    print({'coef_': list(lr.coef_), 'accuracy': accuracy_score(y_test, lr.predict(x_test))})

def stage3():
    x_train, x_test, y_train, y_test = prepare_data(["worst concave points", "worst perimeter", "worst radius"])
    lr = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)
    lr.fit_log_loss(x_train, y_train)
    print({'coef_': list(lr.coef_), 'accuracy': accuracy_score(y_test, lr.predict(x_test))})

def stage4():
    x_train, x_test, y_train, y_test = prepare_data(["worst concave points", "worst perimeter", "worst radius"])

    lr = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)
    result_dict = {}
    d1  = lr.fit_mse(x_train, y_train)
    result_dict['mse_accuracy'] = accuracy_score(y_test, lr.predict(x_test))

    d2 = lr.fit_log_loss(x_train, y_train)
    result_dict['logloss_accuracy'] = accuracy_score(y_test, lr.predict(x_test))

    lr = LogisticRegression(fit_intercept=True,max_iter=1000)
    lr.fit(x_train, y_train)
    result_dict['sklearn_accuracy'] = accuracy_score(y_test, lr.predict(x_test))
    result_dict.update(d1)
    result_dict.update(d2)
    print(result_dict)
    answers = f'''Answers to the questions:
    1) {round(min(result_dict['mse_error_first']),5):.5f}
    2) {round(min(result_dict['mse_error_last']),5):.5f}
    3) {round(max(result_dict['logloss_error_first']),5):.5f}
    4) {round(max(result_dict['logloss_error_last']),5):.5f}
    5) expanded
    6) expanded'''
    print(answers)

if __name__ == '__main__':
    stage4() #stage3() #stage2()  # stage1()
    #stage3()
    #stage2()