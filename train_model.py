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
y_train_log = np.log1p(y_train)
model.fit(X_train, y_train_log)

y_pred_log = model.predict(X_test)

#Domain Constrain
y_pred = np.expm1(y_pred_log)
MSE = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)
R2 = r2_score(y_test, y_pred)

joblib.dump(model, "pokemon_base_exp_model.pkl")