from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score
from datavisualization.matplot_visualisation import matplot_visualisation


class Blobs:
    def __init__(self, n_samples, n_features, center, random_state, test_size):
        self.X, self.Y = make_blobs(n_samples=n_samples, n_features=n_features, centers=center, random_state=random_state)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.X, self.Y, test_size=test_size, random_state=random_state)

    def visualisation(self):
        matplot_visualisation(self.X[:,0], self.X[:, 1], "o", edgecolor="k").visualisation()

    def learning(self):
        model = KNeighborsClassifier(n_neighbors=30)
        model.fit(self.train_x, self.train_y)
        predict = model.predict(self.test_x)
        r2 = r2_score(self.test_y, predict)
        print(round(r2 * 100, 2), "%")
        matplot_visualisation(self.test_y, predict, "X", edgecolor="k").visualisation()


if __name__ == '__main__':
    b = Blobs(250, 2, 6, 20, 0.40)
    # b.visualisation()
    b.learning()