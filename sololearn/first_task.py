import pandas as pd

filename = input()
column_name = input()


pr = pd.read_csv(filename)

df = pr[column_name].values

print(df)