import pandas as pd
import os

path=os.path.join(os.getcwd(),"covid_S3/covid_toy.csv")

df=pd.read_csv(path)
print(df.head())
print(df.isnull().sum())
print(df.dtypes)


df.dropna(inplace=True)

from sklearn.preprocessing import LabelEncoder

lb=LabelEncoder()
df['gender']=lb.fit_transform(df['gender'])
df['cough']=lb.fit_transform(df['cough'])
df['city']=lb.fit_transform(df['city'])
df['has_covid']=lb.fit_transform(df['has_covid'])
print(df.head())

from sklearn.model_selection import train_test_split

X = df.drop("has_covid", axis=1)
y = df["has_covid"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

lr=LogisticRegression()
lr.fit(X_train,y_train)
y_lr=lr.predict(X_test)

dt=DecisionTreeRegressor()
dt.fit(X_train,y_train)
y_dt=dt.predict(X_test)

rf=RandomForestClassifier()
rf.fit(X_train,y_train)
y_rf=rf.predict(X_test)

from sklearn.metrics import accuracy_score
print("lr:",accuracy_score(y_test,y_lr))
print("dt:",accuracy_score(y_test,y_dt))
print("rf:",accuracy_score(y_test,y_rf))

import pickle

with open('covid_S3/data.pkl', 'wb') as f:  # 'wb' = write binary
    pickle.dump(dt, f)
