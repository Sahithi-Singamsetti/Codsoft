# =========================================================
# CUSTOMER CHURN PREDICTION
# =========================================================

# Install required packages if missing:
# pip install pandas numpy scikit-learn xgboost

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================================================
# 1. LOAD DATA
# =========================================================

DATA_PATH = "Churn_Modelling.csv"  # change path if needed
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print(df.head())

# =========================================================
# 2. PREPROCESS DATA
# =========================================================

# Target column
TARGET = "Exited"   # 1 -> churned, 0 -> retained

# Drop unneeded columns (customer id, row identifiers)
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

# Separate features and label
X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\nClass distribution:\n", y.value_counts(normalize=True))

# Identify feature types
categorical_features = [
    "Geography",
    "Gender"
]

numeric_features = [
    c for c in X.columns if c not in categorical_features
]

print("\nNumeric Features:", numeric_features)
print("Categorical Features:", categorical_features)


# =========================================================
# 3. PREPROCESSING PIPELINE
# =========================================================

# OneHotEncode categorical values, scale numeric ones
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# =========================================================
# 4. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================================================
# 5. TRAIN MODELS
# =========================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(
        learning_rate=0.1,
        n_estimators=300,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}

for name, model in models.items():

    print("\n==============================")
    print("Training:", name)

    # Create full pipeline
    clf = Pipeline(steps=[("preprocess", preprocess),
                        ("model", model)
                       ])

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))


# =========================================================
# 6. CUSTOM PREDICTION FUNCTION
# =========================================================

def predict_churn(sample, trained_model):
    """
    sample: dict with feature names as keys
    trained_model: one of the trained pipelines above
    """
    sample_df = pd.DataFrame([sample])
    result = trained_model.predict(sample_df)
    return ("Churn" if result[0] == 1 else "Retained")


# Example usage
example_customer = {
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Female",
    "Age": 45,
    "Tenure": 3,
    "Balance": 50000,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 60000
}

# You can use the last trained model (XGBoost) like this:
# print("Prediction:", predict_churn(example_customer, clf))
