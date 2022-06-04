import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

random_state = int(input())
n = int(input())
rows = []
for i in range(n):
    rows.append([float(a) for a in input().split()])

X = np.array(rows)
y = np.array([int(a) for a in input().split()])

xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state=random_state)

rfc = RandomForestClassifier(n_estimators=5, random_state=random_state)

rfc.fit(xtrain, ytrain)

predict = rfc.predict(xtest)

print(predict)