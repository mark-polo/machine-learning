import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from blitz_tests.blitz_test_for_regression import Blitz


class Mobile_Prices:
    def __init__(self):
        self.df_train = pd.read_csv("datasets/train_for_mobile_prieces.csv")

    def data_analysis(self):
        print("Isna : ", self.df_train.isna().sum())
        print("Isnull : ", self.df_train.isnull().sum())
        print("Desc : ", self.df_train.describe())
        print("Head : ", self.df_train.head())

    def battery_visual_hist(self):
        plt.figure(figsize=(10, 5))
        plt.hist(self.df_train["battery_power"], color="green", edgecolor="black", log=True)
        plt.xlabel("battery_power")
        plt.show()

    def px_height_visual_hist(self):
        plt.figure(figsize=(10, 5))
        plt.hist(self.df_train["px_height"], color="green", edgecolor="black", log=True)
        plt.xlabel("px_height")
        plt.show()

    def px_visual_plot(self):
        X_not_sorted = np.array(self.df_train["px_width"])
        Y_not_sorted = np.array(self.df_train["px_height"])

        X = np.sort(X_not_sorted, kind="mergesort")
        Y = np.sort(Y_not_sorted, kind="mergesort")

        plt.figure(figsize=(10, 5))
        plt.title("Px width and height")
        plt.plot(X, Y, color='r')
        plt.xlabel("px_width")
        plt.ylabel("px_height")
        plt.show()

    def battery_ram_visual_plot(self):
        X_not_sorted = np.array(self.df_train["battery_power"])
        Y_not_sorted = np.array(self.df_train["ram"])

        X = np.sort(X_not_sorted, kind="mergesort")
        Y = np.sort(Y_not_sorted, kind="mergesort")

        plt.figure(figsize=(10, 5))
        plt.title("battery_power and ram")
        plt.plot(X, Y, color='g')
        plt.xlabel("battery_power")
        plt.ylabel("ram")
        plt.show()

    def data_preprocessing(self):
        self.df_train["int_memory"] = self.df_train["int_memory"] * 1000

    def blitz_learning(self, test_size=0.4, random_state=42):
        X = self.df_train.drop(["price_range"], axis=1)
        Y = self.df_train["price_range"]
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        b = Blitz(x_train, x_test, y_train, y_test, neighbor=2, num_folds=20, random_state=0, max_depth=20)

        print(b.testing_models("r2"))

    def learning_rfr_algo(self, test_size=0.4, random_state=42, max_depth=20):
        X = self.df_train.drop(["price_range"], axis=1)
        Y = self.df_train["price_range"]
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        model = RandomForestRegressor(random_state=random_state, max_depth=max_depth)

        model.fit(x_train, y_train)
        predict = model.predict(x_test)
        r2 = r2_score(y_test, predict)

        print(round(r2 * 100, 2), "%")

        self.xy_visual_plot(y_test, predict)

    def xy_visual_plot(self, ytest, pred):
        X_not_sorted = np.array(self.df_train.drop(["price_range"], axis=1))[:, -1]
        Y_not_sorted = np.array(self.df_train["price_range"])

        X = np.sort(X_not_sorted, kind="mergesort")
        Y = np.sort(Y_not_sorted, kind="mergesort")

        Ytest_not_sorted = np.array(ytest)
        Predict_not_sorted = np.array(pred)

        Ytest = np.sort(Ytest_not_sorted, kind="mergesort")
        Predict = np.sort(Predict_not_sorted, kind="mergesort")

        plt.figure(figsize=(10, 5))
        plt.plot(X, Y, 'b', Ytest, Predict, 'c')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Real and Predict")
        plt.show()


if __name__ == '__main__':
    mp = Mobile_Prices()
    # mp.data_analysis()
    mp.data_preprocessing()
    # mp.blitz_learning()
    mp.learning_rfr_algo()