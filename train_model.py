import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

df = pd.read_csv("pokemon_dataset.csv")

X = df.drop(columns = "base_experience")
y = df["base_experience"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Multiple Regression
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
MSE = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)
R2 = r2_score(y_test, y_pred)

joblib.dump(model, "pokemon_base_exp_model.pkl")