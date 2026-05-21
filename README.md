import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    "Day": [1, 2, 3],
    "Temperature": [36, 40, 40]
}

# DataFrame
df = pd.DataFrame(data)

# Input and Output
X = df[["Day"]]
y = df["Temperature"]

# Train Model
model = LinearRegression()
model.fit(X, y)

# Predict Day 4
future_day = [[4]]
prediction = model.predict(future_day)

# Create Figure
plt.figure(figsize=(14,8))

# Real Data
plt.scatter(
    df["Day"],
    df["Temperature"],
    color="blue",
    s=180,
    label="Observed Temperature"
)

# Trend Line
plt.plot(
    df["Day"],
    model.predict(X),
    color="green",
    linewidth=4,
    label="Prediction Trend"
)

# Future Prediction Star
plt.scatter(
    4,
    prediction[0],
    color="red",
    s=450,
    marker="*",
    label="Predicted Day 4"
)

# Prediction Connection Line
plt.plot(
    [3,4],
    [df["Temperature"].iloc[-1], prediction[0]],
    linestyle="--",
    linewidth=3,
    color="orange"
)

# Prediction Value Text
plt.text(
    4,
    prediction[0]+0.5,
    f'{prediction[0]:.2f}°C',
    fontsize=14
)

# Labels
plt.xlabel("Days", fontsize=15)
plt.ylabel("Temperature (°C)", fontsize=15)

# Title
plt.title(
    "Weather Prediction Analysis using Machine Learning",
    fontsize=20
)

# Details Below Graph
plt.figtext(
    0.15, -0.08,
    f"""
Project Details:
• Model Used: Linear Regression
• Input Feature: Day
• Output Prediction: Temperature
• Predicted Day 4 Temperature: {prediction[0]:.2f}°C
• Analysis Type: Forecast Trend Detection
""",
    fontsize=13
)

# Legend
plt.legend(fontsize=13)

# Grid
plt.grid(True)

plt.tight_layout()
plt.show()
