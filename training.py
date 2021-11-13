import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv('hiring.csv')
print(df.head())
print(df.isna().sum())
df['experience'].fillna('0', inplace=True)
df['test_score'].fillna(df['test_score'].mean(), inplace=True)


def fun(exp):
    words = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten':
             10, 'eleven': 11, '0': 0}
    return words[exp]


def map_0(exp):
    if exp == 0:
        return int(df['experience'].median())
    else:
        return exp


df['experience'] = df['experience'].apply(lambda val: fun(val))
df['experience'] = df['experience'].apply(lambda val: map_0(val))
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
model = LinearRegression()
model.fit(X, y)
output = round(model.predict([[3, 7.0, 6]])[0], 2)
print(output)

joblib.dump(model, 'model.pkl')








