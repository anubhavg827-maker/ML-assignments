import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. LOAD THE DATASET
pd.set_option("display.max_columns", 8)
df = pd.read_csv("weather.csv")
print("Shape (rows, columns):", df.shape)
print(df.head())

# 2. CLEAN THE DATA
print("\nMissing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])

num_cols = df.select_dtypes(include="number").columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
cat_cols = df.select_dtypes(exclude="number").columns
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

# 3. BASIC ANALYSIS
print("\nSummary statistics:")
print(df[["MinTemp", "MaxTemp", "Rainfall", "Humidity3pm"]].describe().round(2))
print("\nAverage MaxTemp:", round(df["MaxTemp"].mean(), 2), "C")
print("Hottest day  :", df["MaxTemp"].max(), "C")
print("Coldest day  :", df["MinTemp"].min(), "C")
print("Total rainfall:", round(df["Rainfall"].sum(), 1), "mm")
print("\nRainTomorrow counts:")
print(df["RainTomorrow"].value_counts())

# 4. CHARTS
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
ax[0, 0].plot(df["MaxTemp"], color="red", label="MaxTemp")
ax[0, 0].plot(df["MinTemp"], color="blue", label="MinTemp")
ax[0, 0].set_title("Temperature over the days")
ax[0, 0].set_xlabel("Day")
ax[0, 0].set_ylabel("Temperature (C)")
ax[0, 0].legend()

ax[0, 1].hist(df["Rainfall"], bins=20, color="skyblue", edgecolor="black")
ax[0, 1].set_title("Rainfall distribution")
ax[0, 1].set_xlabel("Rainfall (mm)")

ax[1, 0].scatter(df["Humidity3pm"], df["Temp3pm"], c="green", alpha=0.6)
ax[1, 0].set_title("Humidity vs Temperature (3pm)")
ax[1, 0].set_xlabel("Humidity 3pm (%)")
ax[1, 0].set_ylabel("Temp 3pm (C)")

df["RainTomorrow"].value_counts().plot(kind="bar", ax=ax[1, 1], color=["orange", "purple"])
ax[1, 1].set_title("Rain Tomorrow: No vs Yes")
ax[1, 1].set_xlabel("")
ax[1, 1].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig("weather_charts.png", dpi=110)

# 5. MODEL TRAINING - predict if it will rain tomorrow
# RISK_MM is dropped because it is tomorrow's rain amount (it leaks the answer)
X = df.drop(columns=["RainTomorrow", "RISK_MM"])
X = pd.get_dummies(X, drop_first=True)
y = df["RainTomorrow"].map({"No": 0, "Yes": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTraining rows:", len(X_train), "| Testing rows:", len(X_test))

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

# 6. PREDICT AND EVALUATE
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("\n" + name)
    print("Accuracy:", round(accuracy_score(y_test, pred) * 100, 2), "%")
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))

# 7. SAMPLE PREDICTION
rf = models["Random Forest"]
sample = X_test.iloc[[0]]
print("\nSample prediction for first test day:",
      "Rain" if rf.predict(sample)[0] == 1 else "No Rain",
      "| Actual:", "Rain" if y_test.iloc[0] == 1 else "No Rain")
