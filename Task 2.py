# =========================================================
# CREDIT CARD FRAUD DETECTION
# =========================================================

# Install required packages if missing:
# pip install pandas scikit-learn imbalanced-learn

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Imbalanced dataset balancing
from imblearn.over_sampling import SMOTE

# =========================================================
# 1. LOAD DATA
# =========================================================

# Change this to the path where you downloaded the dataset
DATA_PATH = "fraudTrain.csv"

df = pd.read_csv(DATA_PATH)
print("Dataset Shape:", df.shape)
print(df.head())


# =========================================================
# 2. PREPROCESS
# =========================================================

# Usually "isFraud" is the target label
TARGET = "isFraud"

# Drop rows missing values (if any)
df = df.dropna()

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\nPositive (Fraud) Ratio:", y.sum() / len(y))


# =========================================================
# 3. TRAIN/TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


# =========================================================
# 4. HANDLE SCALE & IMBALANCE
# =========================================================

# Scaling numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE to oversample minority class
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)


print("\nBalanced class counts:", np.bincount(y_train_balanced))


# =========================================================
# 5. TRAIN MODELS
# =========================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

for name, model in models.items():
    print("\n==============================")
    print("Training:", name)

    model.fit(X_train_balanced, y_train_balanced)

    preds = model.predict(X_test_scaled)

    print("Accuracy :", accuracy_score(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))


# =========================================================
# 6. TEST CUSTOM TRANSACTION
# =========================================================

# Example transaction (use your own values instead)
sample_transaction = np.array([X_test_scaled[0]])  # test sample

print("\nSample Prediction (0 Legitimate, 1 Fraud):",
      models["Random Forest"].predict(sample_transaction)[0])

