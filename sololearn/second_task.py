from sklearn.linear_model import LogisticRegression
import numpy as np

model = LogisticRegression()

n = int(input())
X = []
for i in range(n):
    X.append([float(x) for x in input().split()])
y = [int(x) for x in input().split()]
datapoint = [float(x) for x in input().split()]

datapoint = np.array(datapoint).reshape(1, -1)

model.fit(X, y)
print(model.predict(datapoint[[0]])[0])