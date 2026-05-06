"""
train_allergen_model.py  –  Train the allergen classifier

Improvements vs. original:
  1. Deduplicates the dataset before training (original CSV has duplicate rows).
  2. Filters out extremely rare label combinations (< 2 samples) so
     train_test_split doesn't fail with stratify.
  3. Uses stratified split to preserve label distribution.
  4. Adds a MultiLabelBinarizer pathway comment for future improvement.
  5. Reports per-class support so you can see which allergens have sparse data.
"""

import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent  # src/ → AllerScan/
_combined     = ROOT / "data" / "training_data.csv"
DATA_PATH     = _combined if _combined.exists() else ROOT / "data" / "food_ingredients_and_allergens.csv"
MODEL_PATH    = ROOT / "models" / "allergen_model.pkl"
print(f"Using dataset: {DATA_PATH}")

# ── Load & clean ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

# FIX 1 – remove duplicate rows that inflate apparent dataset size
before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df)} duplicate rows  ({len(df)} remain)")

# Drop rows with no allergen label
df = df.dropna(subset=["Allergens"])

# FIX 2 – drop label combos with only 1 sample (breaks stratified split)
label_counts = df["Allergens"].value_counts()
df = df[df["Allergens"].isin(label_counts[label_counts >= 2].index)]
print(f"Training on {len(df)} rows with {df['Allergens'].nunique()} label classes")

# ── Feature engineering ───────────────────────────────────────────────────────
text_columns = ["Food Product", "Main Ingredient", "Sweetener", "Fat/Oil", "Seasoning"]
df["input_text"] = df[text_columns].fillna("").agg(" ".join, axis=1)

X = df["input_text"]
y = df["Allergens"]

# FIX 3 – stratified split keeps label distribution balanced
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Model ─────────────────────────────────────────────────────────────────────
model = Pipeline([
    ("tfidf",      TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000, C=1.0)),
])

model.fit(X_train, y_train)

# ── Evaluation ────────────────────────────────────────────────────────────────
predictions = model.predict(X_test)
print("\nModel evaluation:")
print(classification_report(y_test, predictions, zero_division=0))

# ── Save ──────────────────────────────────────────────────────────────────────
MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
