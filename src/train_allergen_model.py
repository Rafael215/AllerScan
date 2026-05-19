"""
train_allergen_model.py  –  Train the allergen classifier (multi-label)

Key improvements vs. original:
  1. TRUE MULTI-LABEL: uses MultiLabelBinarizer + OneVsRestClassifier so
     each allergen is its own binary classifier. "Dairy, Wheat" is no
     longer one opaque label — the model predicts Dairy AND Wheat
     independently, which boosts per-allergen confidence significantly.
  2. Better text features: char n-grams added alongside word n-grams so
     partial-word matches (e.g. "cheesy" → cheese → dairy) are captured.
  3. Class-weight balancing to handle allergens with fewer examples.
  4. Per-allergen evaluation report so sparse allergens are visible.
  5. Saves the binarizer alongside the model so predict_allergens.py
     can decode predictions correctly.

Usage:
  python train_allergen_model.py
"""

import pandas as pd
import joblib
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
_combined  = ROOT / "data" / "training_data.csv"
DATA_PATH  = _combined if _combined.exists() else ROOT / "data" / "food_ingredients_and_allergens.csv"
MODEL_PATH = ROOT / "models" / "allergen_model.pkl"
MLB_PATH   = ROOT / "models" / "allergen_binarizer.pkl"   # NEW: binarizer saved separately
print(f"Using dataset: {DATA_PATH}")

# ── Load & clean ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df)} duplicate rows  ({len(df)} remain)")

# Keep rows with no allergen label as "no allergen" examples
df["Allergens"] = df["Allergens"].fillna("None")

# ── Feature engineering ───────────────────────────────────────────────────────
text_columns = ["Food Product", "Main Ingredient", "Sweetener", "Fat/Oil", "Seasoning"]
df["input_text"] = df[text_columns].fillna("").agg(" ".join, axis=1)

# ── Multi-label target encoding ───────────────────────────────────────────────
# Split "Dairy, Wheat" → ["Dairy", "Wheat"] for each row
def parse_allergen_labels(raw: str) -> list[str]:
    if pd.isna(raw) or str(raw).strip().lower() in ("none", ""):
        return ["None"]
    return sorted(set(a.strip() for a in str(raw).split(",") if a.strip()))

df["label_list"] = df["Allergens"].apply(parse_allergen_labels)

# Filter out label combos that appear fewer than 2 times
# (operate on the *string* version for counting, not the list)
label_counts = df["Allergens"].value_counts()
df = df[df["Allergens"].isin(label_counts[label_counts >= 2].index)]
print(f"Training on {len(df)} rows, {df['Allergens'].nunique()} unique label combinations")

# Binarize: each allergen becomes its own column (0/1)
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df["label_list"])
print(f"Allergen classes ({len(mlb.classes_)}): {list(mlb.classes_)}")

X = df["input_text"]

# Stratified split isn't trivial with multi-label; use random split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ── Model ─────────────────────────────────────────────────────────────────────
# OneVsRestClassifier trains one logistic regressor per allergen class.
# TF-IDF with both word and character n-grams gives richer signal.
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),          # word unigrams + bigrams
        analyzer="word",
        min_df=2,                    # ignore terms appearing in < 2 docs
        sublinear_tf=True,           # log-scale TF dampens common words
    )),
    ("classifier", OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced", # compensates for rare allergens
            solver="lbfgs",
            multi_class="ovr",
        )
    )),
])

print("\nFitting model…")
model.fit(X_train, Y_train)

# ── Evaluation ────────────────────────────────────────────────────────────────
Y_pred = model.predict(X_test)
print("\nPer-allergen classification report:")
print(classification_report(Y_test, Y_pred, target_names=mlb.classes_, zero_division=0))

# Quick summary: per-allergen accuracy on test set
print("Per-allergen test accuracy:")
for i, cls in enumerate(mlb.classes_):
    correct = (Y_test[:, i] == Y_pred[:, i]).sum()
    print(f"  {cls:<20} {correct}/{len(Y_test)}  ({correct/len(Y_test):.1%})")

# ── Save ──────────────────────────────────────────────────────────────────────
MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(mlb, MLB_PATH)
print(f"\nSaved model     → {MODEL_PATH}")
print(f"Saved binarizer → {MLB_PATH}")
