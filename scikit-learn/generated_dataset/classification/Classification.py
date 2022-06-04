from datavisualization.matplot_visualisation import matplot_visualisation
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import numpy as np


class Classification:
    def __init__(self, test_size, random_state):
        self.X, self.Y = make_classification(n_samples=120, n_features=20, n_classes=2, random_state=random_state)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.X, self.Y, test_size=test_size,
                                                                                random_state=random_state)

    def learning_model(self):
        print(np.array(self.train_x).shape)  # (70, 20)
        print(np.array(self.train_y).shape)  # (70,)

        model = make_pipeline(StandardScaler(), SVC())
        model.fit(self.train_x, self.train_y)
        predict = model.predict(self.test_x)
        metric = round(metrics.accuracy_score(self.test_y, predict) * 100, 2)

        print(f"{metric} %")
        print(metrics.classification_report(self.test_y, predict))

        print(predict)
        print(self.train_y) 

        matplot_visualisation(self.X[:, -1], self.Y).visualisation()


if __name__ == '__main__':
    clf = Classification(0.40, 42)
    clf.learning_model()