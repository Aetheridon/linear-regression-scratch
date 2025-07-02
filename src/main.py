from linear_regression import LinearRegression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

# Scaling
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()

model = LinearRegression(X_train, y_train, learning_rate=0.01)
model.train(epochs=100)

y_pred = model.predict(X_test)

plot_df = pd.DataFrame({
    'Size_cm': X_test,
    'Actual_Weight_g': y_test,
    'Predicted_Weight_g': y_pred
})

for pred, actual in zip(y_pred, y_test):
    print(f"Predicted: {pred:.2f}, Actual: {actual}")

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Size_cm', y='Actual_Weight_g', data=plot_df, label='Actual', color='blue')
sns.scatterplot(x='Size_cm', y='Predicted_Weight_g', data=plot_df, label='Predicted', color='orange', marker='X')
sns.lineplot(x='Size_cm', y='Predicted_Weight_g', data=plot_df, color='red', label='Regression Line')

plt.title('Actual vs Predicted Mouse Weights')
plt.xlabel('Size (cm)')
plt.ylabel('Weight (g)')
plt.legend()
plt.tight_layout()
plt.savefig("model.png")