import pandas as pd

data = pd.read_csv("worker_dataset.csv")

data = data.drop(["Count", "Year"], axis=1)

worker = data["Worker Name"]

seasons = pd.get_dummies(data["Season"], prefix="Season")


if __name__ == '__main__':
    print(seasons)