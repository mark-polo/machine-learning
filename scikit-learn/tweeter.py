import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier


class Tweets:
    def __init__(self):
        self.df = pd.read_csv("datasets/Tweets.csv")

    def learning(self):
        X = self.df["selected_text"].values.astype('U') # The astype(‘U’) is telling numpy to convert the data to Unicode (essentially a string in python 3).
        Y = self.df["sentiment"].values.astype('U')
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=10)

        model = make_pipeline(TfidfVectorizer(lowercase=True, stop_words='english', analyzer='word'),
                              SGDClassifier(loss='hinge', penalty='l2', random_state=42))

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.fit_transform(y_test)

        print(x_train)
        print(y_train)

        model.fit(x_train, y_train)
        predict = model.predict(x_test)
        print(round(accuracy_score(predict, y_test) * 100, 2), " %")


if __name__ == '__main__':
    t = Tweets()
    t.learning()