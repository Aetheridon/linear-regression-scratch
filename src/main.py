from linear_regression import LinearRegression
import pandas as pd

df = pd.read_csv("mice_data.csv")

def split(df: pd.DataFrame, train_percent):
    val = int(len(df) * (train_percent / 100))
    train_df = df.iloc[:val,:]
    test_df = df.iloc[val:,:]

    X_train, y_train = train_df["Size_cm"], train_df["Weight_g"]
    X_test, y_test = test_df["Size_cm"], test_df["Weight_g"]

    return (X_train, y_train, X_test, y_test)

train_test = split(df, 80)
X_train, y_train, X_test, y_test = train_test

model = LinearRegression(X_train, y_train, learning_rate=0.01)
model.train(epochs=100)

y_pred = model.predict(X_test)

for pred, actual in zip(y_pred, y_test):
    print(f"Predicted: {pred:.2f}, Actual: {actual}")