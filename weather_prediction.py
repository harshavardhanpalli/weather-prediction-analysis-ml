import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    "Day": [1, 2, 3],
    "Temperature": [36, 40, 40]
}

df = pd.DataFrame(data)

X = df[["Day"]]
y = df["Temperature"]

model = LinearRegression()
model.fit(X, y)

future_day = [[4]]
prediction = model.predict(future_day)

print("Predicted Temperature for Day 4:", prediction[0])

plt.figure(figsize=(14,8))

plt.scatter(
    df["Day"],
    df["Temperature"],
    color="blue",
    s=180,
    label="Observed Temperature"
)

plt.plot(
    df["Day"],
    model.predict(X),
    color="green",
    linewidth=4,
    label="Prediction Trend"
)

plt.scatter(
    4,
    prediction[0],
    color="red",
    s=450,
    marker="*",
    label="Predicted Day 4"
)

plt.plot(
    [3,4],
    [df["Temperature"].iloc[-1], prediction[0]],
    linestyle="--",
    linewidth=3,
    color="orange"
)

plt.text(
    4,
    prediction[0]+0.5,
    f'{prediction[0]:.2f}°C',
    fontsize=14
)

plt.xlabel("Days", fontsize=15)
plt.ylabel("Temperature (°C)", fontsize=15)

plt.title(
    "Weather Prediction Analysis using Machine Learning",
    fontsize=20
)

plt.legend(fontsize=13)
plt.grid(True)

plt.tight_layout()
plt.show()
