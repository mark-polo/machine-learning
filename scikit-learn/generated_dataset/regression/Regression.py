from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from datavisualization.matplot_visualisation import matplot_visualisation


class Regression:
    def __init__(self, test_size, random_state):
        self.X, self.Y = make_regression(n_samples=500, n_features=1, noise=10, random_state=random_state)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.X, self.Y, test_size=test_size, random_state=random_state)

    def learning_model(self):
        model = KNeighborsRegressor(n_neighbors=50, weights="distance")
        model.fit(self.train_x, self.train_y)
        predict = model.predict(self.test_x)
        r2 = r2_score(self.test_y, predict)
        print(f"{round(r2 * 100 , 2)} %")

        # matplot_visualisation(self.X, self.Y).visualisation()
        matplot_visualisation(self.test_x,  predict).visualisation()


if __name__ == '__main__':
    rg = Regression(0.30, 42)
    rg.learning_model()