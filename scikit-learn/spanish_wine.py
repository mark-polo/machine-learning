import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from blitz_tests.blitz_test_for_regression import Blitz


class Spanish_wine:
    def __init__(self):
        self.df = pd.read_csv("datasets/wines_SPA.csv")

    def preprocessing_data(self):
        self.df['year'] = self.df['year'].replace('N.V.', np.NaN)
        self.df = self.df.dropna()
        self.df['year'] = self.df['year'].astype(np.int64)
        self.df = self.df.drop(["country"], axis=1)

        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                label = LabelEncoder()
                label = label.fit(self.df[col])
                self.df[col] = label.transform(self.df[col].astype(str))

        self.df = (self.df - self.df.mean()) / self.df.std()  # z = (x - u) / s -> sklearn.preprocessing.StandardScaler

    def blitz_testing_model(self, test_size, random_state):
        X = self.df.drop(["price"], axis=1)
        Y = self.df["price"]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        b = Blitz(X_train, X_test, y_train, y_test, neighbor=2, num_folds=20, random_state=0, max_depth=20)

        print(b.testing_models("r2"))

        self.learning_model(X_train, X_test, y_train, y_test)

    @staticmethod
    def learning_model(x_train, x_test, y_train, y_test):
        model_rfr = RandomForestRegressor(random_state=0, max_depth=20)
        model_rfr.fit(x_train, y_train)
        predict_rfr = model_rfr.predict(x_test)
        r2_rfr = r2_score(y_test, predict_rfr)

        print(f"RFR :  {round(r2_rfr * 100, 2)} %")


if __name__ == '__main__':
    sw = Spanish_wine()
    sw.preprocessing_data()
    sw.blitz_testing_model(0.4, 42)
