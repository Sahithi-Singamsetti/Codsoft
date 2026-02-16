# =========================================================
# MOVIE GENRE CLASSIFICATION MODEL
# TF-IDF + Naive Bayes + Logistic Regression + SVM
# =========================================================

# Install dependencies if not installed
# pip install pandas scikit-learn nltk

import pandas as pd
import numpy as np
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Download stopwords (only first time)
nltk.download('stopwords')
from nltk.corpus import stopwords


# =========================================================
# 1. LOAD DATASET
# =========================================================

# Change path to your downloaded dataset file
DATA_PATH = "IMDb Genre Dataset.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nColumns:", df.columns)
print(df.head())


# =========================================================
# 2. CLEAN DATA
# =========================================================

# Change column names if dataset differs
TEXT_COLUMN = "plot"
LABEL_COLUMN = "genre"

df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])

X = df[TEXT_COLUMN].astype(str)
y = df[LABEL_COLUMN].astype(str)


# =========================================================
# 3. ENCODE LABELS
# =========================================================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nGenres:", list(label_encoder.classes_))


# =========================================================
# 4. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# =========================================================
# 5. TF-IDF VECTORIZATION
# =========================================================

tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1,2),
    stop_words=stopwords.words("english")
)

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)


# =========================================================
# 6. TRAIN MODELS
# =========================================================

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Linear SVM": LinearSVC()
}

trained_models = {}

for name, model in models.items():
    print("\n==============================")
    print("Training:", name)

    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)

    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds,
                                target_names=label_encoder.classes_))

    trained_models[name] = model


# =========================================================
# 7. PREDICTION FUNCTION
# =========================================================

def predict_genre(text, model_name="Linear SVM"):
    model = trained_models[model_name]
    vector = tfidf.transform([text])
    pred = model.predict(vector)
    return label_encoder.inverse_transform(pred)[0]


# =========================================================
# 8. TEST CUSTOM INPUT
# =========================================================

sample_plot = """
A group of astronauts travel through a wormhole in space
to ensure humanity's survival and discover strange new worlds.
"""

print("\nSample Prediction:")
print(predict_genre(sample_plot))
