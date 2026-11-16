import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
df = pd.read_csv("Student_Performance.csv")
df.head()
print(df.shape)                    # (rows, columns)
df.info()                          # column types and non-null counts
print(df.isna().sum())             # missing values per column
print("duplicates:", df.duplicated().sum())
df.describe()  
df = df.drop_duplicates()
df
df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})
print(df["Extracurricular Activities"].value_counts())
print(df.isna().sum().sum(), "missing values")
df
df = df.fillna(df.median(numeric_only=True))
df.head()
df["Performance Index"].hist(bins=20)
plt.xlabel("Performance Index")
plt.ylabel("Number of students")
plt.title("Distribution of Performance Index")
plt.show()
plt.scatter(df["Previous Scores"], df["Performance Index"], alpha=0.3)
plt.xlabel("Previous Scores")
plt.ylabel("Performance Index")
plt.title("Previous Scores vs Performance Index")
plt.show()

print(df.corr()["Performance Index"].sort_values(ascending=False))
from sklearn.model_selection import train_test_split

X = df.drop(columns="Performance Index")   # features
y = df["Performance Index"]                # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.3f}")
from sklearn.dummy import DummyRegressor

baseline = DummyRegressor(strategy="mean")   # always predicts the average
baseline.fit(X_train, y_train)
base_pred = baseline.predict(X_test)

print("Baseline MAE:", round(mean_absolute_error(y_test, base_pred), 2))
print("Model MAE   :", round(mae, 2))
coefs = pd.Series(model.coef_, index=X.columns)
print(coefs.sort_values(ascending=False).round(2))
print("Intercept:", round(model.intercept_, 2))
plt.scatter(y_test, pred, alpha=0.3)
low, high = y_test.min(), y_test.max()
plt.plot([low, high], [low, high], color="black")   # perfect predictions
plt.xlabel("Actual Performance Index")
plt.ylabel("Predicted Performance Index")
plt.title("Predicted vs actual")
plt.show()