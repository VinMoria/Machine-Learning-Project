import pandas as pd

df = pd.read_excel("project/Dataset.xlsx",skiprows=range(3))
df = df.drop(df.columns[0], axis=1)
print(df.columns)

