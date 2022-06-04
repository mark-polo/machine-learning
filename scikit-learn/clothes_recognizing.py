import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics, svm
from sklearn import model_selection


def main():
    data = pd.read_csv("datasets/fashion-mnist_test.csv")
    X = data.drop("label", axis=1)
    Y = data["label"]

    class_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                  'Ankle boot']

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.30, random_state=42)

    X_train = X_train / 255
    X_test = X_test / 255

    model = svm.SVC(gamma=0.001, random_state=10)
    model.fit(X_train, Y_train)
    predict = model.predict(X_test)

    p = np.array(predict)

    print(
        f"Classification report for classifier {model}:\n"
        f"{metrics.classification_report(Y_test, predict)}\n"
    )

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(np.array(X_test.iloc[i]).reshape(28, 28), cmap=plt.cm.binary)
        plt.xlabel(class_name[p[i]])


if __name__ == '__main__':
    main()
    plt.show()