import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

model = datasets.load_digits()


def training_visualisation():
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, model.images, model.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)


def learning():
    # flatten the images
    n_samples = len(model.images)
    data = model.images.reshape(n_samples, -1)

    # clf = svm.SVC(gamma=0.001)

    clf = svm.SVC(gamma=0.001)

    X_train, X_test, y_train, y_test = train_test_split(data, model.target, test_size=0.5, shuffle=False)
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)

    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predict)}\n"
    )

    prediction_visualisation(X_test, predict)


def prediction_visualisation(X_test, predicted):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")


if __name__ == '__main__':
    learning()
    plt.show()