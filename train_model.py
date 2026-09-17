import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

data = pd.read_csv(
    "gesture_data.csv",
    header=None
)

print("Dataset loaded!")
print("Number of samples:", len(data))

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions
    )
)

with open(
    "gesture_model.pkl",
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )

print("\nModel saved as gesture_model.pkl")