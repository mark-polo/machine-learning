import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datavisualization.matplot_visualisation import matplot_visualisation
from blitz_tests.blitz_test_for_regression import Blitz
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


class Loan:
    def __init__(self):
        self.df = pd.read_csv("datasets/original.csv")

    def preprocessing(self):
        self.df["income"] = self.df["income"].replace(",", np.NAN)
        self.df["age"] = self.df["age"].replace(",", np.NAN)
        self.df["loan"] = self.df["loan"].replace(",", np.NAN)

        self.df = self.df.dropna()  # drop every NAN

        self.df["age"] = self.df["age"].astype(np.int64)
        self.df["income"] = self.df["income"].astype(np.int64)
        self.df["loan"] = self.df["loan"].astype(np.int64)

        self.df = self.df.drop(["clientid"], axis=1)

        self.df = (self.df - self.df.mean()) / self.df.std()

    def visualisation(self):
        plt.hist2d(self.df["income"], self.df["loan"])
        plt.show()

    def plot_visualisation(self):
        plt.plot(self.df["income"][:10], self.df["age"][:10])
        plt.show()

    def blitz_learning(self, test_size, random_state):
        X = self.df.drop(["default"], axis=1)
        Y = self.df["default"]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        b = Blitz(X_train, X_test, y_train, y_test, neighbor=2, num_folds=20, random_state=0, max_depth=20)

        print(b.testing_models("r2"))

    def learning(self, test_size, random_state):
        X = self.df.drop(["default"], axis=1)
        Y = self.df["default"]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        model = RandomForestRegressor(random_state=0, max_depth=20)

        model.fit(X_train, y_train)

        predict = model.predict(X_test)

        r2 = r2_score(y_test, predict)

        print("RFR : ", round(r2 * 100, 2), " %")


if __name__ == '__main__':
    l = Loan()
    l.preprocessing()
    l.learning(0.4, 42)
    l.plot_visualisation()