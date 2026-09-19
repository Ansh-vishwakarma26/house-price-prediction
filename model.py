import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score 

df = pd.read_csv("Housing.csv")

df.head()
df.info()
df.describe()

x = df.drop(["price"], axis=1)
y = df["price"]

x.head()
y.head()

x.shape
y.shape

print(df["mainroad"].value_counts())
print(df["furnishingstatus"].value_counts())

df["mainroad"]= df["mainroad"].map({'yes':1, 'no':0})
df["guestroom"]= df["guestroom"].map({'yes':1, 'no':0})
df["basement"]= df["basement"].map({'yes':1, 'no':0})
df["hotwaterheating"]= df["hotwaterheating"].map({'yes':1, 'no':0})
df["airconditioning"]= df["airconditioning"].map({'yes':1, 'no':0})
df["prefarea"]= df["prefarea"].map({'yes':1, 'no':0})
df["furnishingstatus"]= df["furnishingstatus"].map({'unfurnished':0,'semi-furnished':1,'furnished':2})

print(df.head())

x = df.drop(["price"], axis=1)
y = df["price"]

print(x.head())

#Train Test Split
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=10)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#Linear Regression model
model = LinearRegression()

model.fit(x_train,y_train)

#Now predict
predictions = model.predict(x_test)

print(predictions)

# to find mean absolute error

mae = mean_absolute_error(y_test, predictions)
print(mae)

# to find r squared score

r2 = r2_score(y_test, predictions)
print(r2)

joblib.dump(model,"model.pkl")