from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

"""
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        bc = Blitz(x_train, x_test, y_train, y_test, 200, 10)
        
        print(bc.testing_models("accuracy"))
"""


class Blitz:
    def __init__(self, X_train, X_test, Y_train, Y_test, n_estimator, num_folds):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.n_estimator = n_estimator
        self.num_folds = num_folds
        # self.seed = seed

    def testing_models(self, scoring):
        models = [('LR', LogisticRegression()),
                  ('LDA', LinearDiscriminantAnalysis()),
                  ('KNN', KNeighborsClassifier()),
                  ('CART', DecisionTreeClassifier()),
                  ('NB', GaussianNB()),
                  ('LSVC', LinearSVC()),
                  ('SVC', SVC()),
                  ('MLP', MLPClassifier()),
                  ('BG', BaggingClassifier(n_estimators=self.n_estimator)),
                  ('RF', RandomForestClassifier(n_estimators=self.n_estimator)),
                  ('ET', ExtraTreesClassifier(n_estimators=self.n_estimator)),
                  ('AB', AdaBoostClassifier(n_estimators=self.n_estimator, algorithm='SAMME')),
                  ('GB', GradientBoostingClassifier(n_estimators=self.n_estimator))]

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
            m_score = model.score(self.X_test, self.Y_test)
            scores.append(m_score)
            msg = "%s: train = %.3f (%.3f) / test = %.3f" % (name, cv_results.mean(), cv_results.std(), m_score)
            msg_row.append(msg)
            print(msg)