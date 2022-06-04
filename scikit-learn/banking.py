import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from blitz_tests.blitz_test_for_classification import Blitz


class Banking:
    def __init__(self):
        self.df_train = pd.read_csv("datasets/train_for_banking.csv", sep=";")

    def data_analysis(self):
        print(self.df_train.isna().sum())

    def data_visual_hist(self):
        plt.figure(figsize=(5, 5))
        plt.hist(self.df_train["balance"], color="Blue", edgecolor="Black", log=True)
        plt.xlabel("balance")
        plt.show()

    def data_visual_plot(self):
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.df_train["balance"], self.df_train["age"])
        plt.show()

    def data_visual_scatter(self):
        fig, ax = plt.subplots(1, 1)
        ax.scatter(self.df_train["balance"], self.df_train["age"])
        plt.show()

    def data_preprocessing(self):
        self.df_train = self.df_train.drop(["contact"], axis=1)

        for col in self.df_train.columns:
            if self.df_train[col].dtype == 'object':
                label = LabelEncoder()
                label = label.fit(self.df_train[col])
                self.df_train[col] = label.transform(self.df_train[col])

        print(self.df_train["y"])

    def blitz_learning(self, test_size, random_state):
        X = self.df_train.drop(["y"], axis=1)
        Y = self.df_train["y"]
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        bt = Blitz(x_train, x_test, y_train, y_test, 200, 10)
        print(bt.testing_models("accuracy"))


if __name__ == '__main__':
    b = Banking()
    b.data_preprocessing()
    # b.data_visual_scatter()
    b.blitz_learning(0.4, 42)
