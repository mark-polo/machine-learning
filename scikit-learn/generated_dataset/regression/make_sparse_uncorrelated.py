from datavisualization.matplot_visualisation import matplot_visualisation
from sklearn.datasets import make_sparse_uncorrelated
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


class MSU:
    def __init__(self, random_state, test_size):
        self.X, self.Y = make_sparse_uncorrelated(1000, 4, random_state=random_state)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.X, self.Y, test_size=test_size,
                                                                                random_state=random_state)

    def visualisation(self):
        matplot_visualisation(self.X[:, -1], self.Y).visualisation()

    def learning(self):
        model = KNeighborsRegressor(n_neighbors=10)
        model.fit(self.train_x, self.train_y)
        predict = model.predict(self.test_x)
        r2 = r2_score(self.test_y, predict)
        print("R2 score : ", round(r2 * 100, 2), "%")


if __name__ == '__main__':
    msu = MSU(42, 0.33)
    msu.learning()
    # msu.visualisation()