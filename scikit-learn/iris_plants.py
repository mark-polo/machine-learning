import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn import metrics

'''
Accuracy score is only for classification problems. 
For regression problems you can use: R2 Score, MSE (Mean Squared Error), RMSE (Root Mean Squared Error).


Slicing with [:, :-1] will give you a 2-dimensional array (including all rows and all columns excluding the last column).

Slicing with [:, 1] will give you a 1-dimensional array (including all rows from the second column). To make this array also 2-dimensional use [:, 1:2] or [:, 1].reshape(-1, 1) or [:, 1][:, None] instead of [:, 1]. This will make x and y comparable.

An alternative to making both arrays 2-dimensional is making them both one dimensional. For this one would do [:, 0] (instead of [:, :1]) for selecting the first column and [:, 1] for selecting the second column.
'''


def data_analysis():
    df = load_iris(as_frame=True)
    data = df['data']
    target = df['target']
    X = data
    Y = target

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.50, random_state=42)

    return np.array([X, Y, train_x, test_x, train_y, test_y])


def training_and_predicting() -> None:

    X_train = np.array(data_analysis()[2])
    X_test = np.array(data_analysis()[3])
    Y_train = np.array(data_analysis()[4])
    Y_test = np.array(data_analysis()[5])

    model = RandomForestClassifier(n_estimators=30, max_depth=30)
    model.fit(X_train, Y_train)
    predict = model.predict(X_test)
    acc = metrics.accuracy_score(Y_test, predict)

    print(f"{model.feature_importances_} \n"
          f"{round(acc * 100, 2)} %")

    # data_visualisation(X_train[:, 1], predict, color='blue')


def data_visualisation(X, Y, color) -> None:
    plt.scatter(X, Y, color=color)
    plt.show()


if __name__ == '__main__':
    data_analysis()
    training_and_predicting()