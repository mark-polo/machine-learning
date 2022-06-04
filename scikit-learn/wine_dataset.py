from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


def learning(test_size, random_state):
    x, y = load_wine(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    pipeline = make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB())

    pipeline.fit(x_train, y_train)

    predict = pipeline.predict(x_test)

    return f"Prediction : {round(accuracy_score(y_test, predict) * 100, 2)} %"


if __name__ == '__main__':
    print(learning(0.33, 30))
