import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from blitz_tests.blitz_test_for_classification import Blitz
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer


class Salary:
    def __init__(self):
        self.df = pd.read_csv("datasets/salary.csv")

    def data_analysis(self):
        mean_age = self.df["age"].mean()
        mean_salary_per_hours = self.df["hours-per-week"].mean()
        print("Mean age : ", mean_age)
        print("Mean salary : ", mean_salary_per_hours)

    def data_visualisation_workclass(self):
        plt.figure(figsize=(10,10))
        self.df["workclass"].value_counts().plot(kind="bar", color="green")  # this staff using when data type is a string
        plt.xlabel("Workclass")
        plt.tight_layout()
        plt.show()

    def data_visualisation_fnlwgt(self):
        plt.figure(figsize=(10,10))
        plt.hist(self.df["fnlwgt"], color="Blue", edgecolor="Black", log=True) # this staff using when data type is a int
        plt.tight_layout()
        plt.xlabel("Fnlwgt")
        plt.show()

    def data_preprocessing(self):
        self.df = self.df.dropna(how="any")
        salary = LabelBinarizer()
        self.df["salary"] = salary.fit_transform(self.df["salary"])

        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                label = LabelEncoder()
                label = label.fit(self.df[col])
                self.df[col] = label.transform(self.df[col])

    def blitz_learning(self, test_size, random_state):
        X = self.df.drop(["salary"], axis=1)
        Y = self.df["salary"]

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        bt = Blitz(x_train, x_test, y_train, y_test, 100, 10)
        print(bt.testing_models('accuracy'))


if __name__ == '__main__':
    s = Salary()
    # s.data_analysis()
    s.data_preprocessing()
    # s.blitz_learning(0.4, 42)
    s.data_visualisation_fnlwgt()
