from sklearn.linear_model import LinearRegression, Lasso, Ridge, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

"""
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        br = Blitz(X_train, X_test, y_train, y_test, neighbor=2, num_folds=20, random_state=0, max_depth=20)

        print(br.testing_models("r2"))
"""


class Blitz:
    def __init__(self, X_train, X_test, Y_train, Y_test, neighbor, num_folds, random_state, max_depth):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.neighbor = neighbor
        self.num_folds = num_folds
        self.random_state = random_state
        self.max_depth = max_depth

    def testing_models(self, scoring):
        models = [('LR', LinearRegression()),
                  ('LA', Lasso()),
                  ('RID', Ridge()),
                  ('BR', BayesianRidge()),
                  ('DTR', DecisionTreeRegressor()),
                  ('LSVR', LinearSVR()),
                  ('KNR', KNeighborsRegressor(n_neighbors=self.neighbor)),
                  ('RFR', RandomForestRegressor(random_state=self.random_state, max_depth=self.max_depth))]

        scores = []
        names = []
        results = []
        predictions = []
        msg_row = []
        for name, model in models:
            kfold = KFold(n_splits=self.num_folds)
            cv_results = cross_val_score(model, self.X_train, self.Y_train, cv=kfold, scoring=scoring)
            names.append(name)
            results.append(cv_results)
            model.fit(self.X_train, self.Y_train)
            m_predict = model.predict(self.X_test)
            predictions.append(m_predict)
            r2 = r2_score(self.Y_test, m_predict)
            scores.append(r2)
            msg = "%s: train = %.3f (%.3f) / test = %.2f" % (name, cv_results.mean(), cv_results.std(), r2 * 100)
            msg_row.append(msg)
            print(msg)