"""
build_training_data.py  –  Convert restaurant-menus.csv into labeled training data

Pipeline:
  1. Load restaurant-menus.csv (Uber Eats dataset)
  2. Auto-label each item using allergen_database.csv keyword matching
  3. Normalise all allergen labels to a single canonical vocabulary
  4. Merge with existing food_ingredients_and_allergens.csv labels
  5. Output a unified training CSV ready for train_allergen_model.py

Usage:
  python build_training_data.py
  python build_training_data.py --menus path/to/restaurant-menus.csv
  python build_training_data.py --limit 50000   # cap rows for speed
"""

import re
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
MENUS_PATH    = ROOT / "restaurant-menus.csv"
DB_PATH       = ROOT / "allergen_database.csv"
EXISTING_PATH = ROOT / "data" / "food_ingredients_and_allergens.csv"
OUTPUT_PATH   = ROOT / "data" / "training_data.csv"

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--menus",  default=str(MENUS_PATH), help="Path to restaurant-menus.csv")
parser.add_argument("--limit",  type=int, default=None,  help="Max rows to load from menus CSV")
parser.add_argument("--output", default=str(OUTPUT_PATH),help="Output training CSV path")
args = parser.parse_args()

# ── Canonical allergen name map ───────────────────────────────────────────────
# Maps allergen_database group_names → unified display labels used in training.
CANONICAL: dict[str, str] = {
    "milk":      "Dairy",
    "egg":       "Eggs",
    "peanut":    "Peanuts",
    "tree_nut":  "Tree Nuts",
    "soy":       "Soybeans",
    "wheat":     "Wheat",
    "gluten":    "Wheat",        # treat as same class
    "fish":      "Fish",
    "shellfish": "Shellfish",
    "sesame":    "Sesame",
    "mustard":   "Mustard",
    "celery":    "Celery",
    "corn":      "Corn",
    "coconut":   "Coconut",
    "garlic":    "Garlic",
    "onion":     "Onion",
    "legume":    "Legumes",
}

# Normalise messy labels already present in the existing CSV
EXISTING_LABEL_FIX: dict[str, str] = {
    "Milk":             "Dairy",
    "Dairy, Anchovies": "Dairy, Fish",
    "Dairy, Ghee":      "Dairy",
    "Fish, Eggs":       "Eggs, Fish",
    "Almonds, Wheat, Dairy": "Dairy, Tree Nuts, Wheat",
}


def canonicalise_group(raw: str) -> str:
    """Map a raw allergen_db group_name to its canonical display label."""
    return CANONICAL.get(raw, raw.title())


def normalise_existing_label(label: str) -> str:
    """Fix known bad labels in the existing CSV and re-sort components."""
    if pd.isna(label):
        return label
    label = EXISTING_LABEL_FIX.get(label, label)
    # Re-sort comma-separated components so "Wheat, Dairy" == "Dairy, Wheat"
    parts = sorted(set(p.strip() for p in label.split(",")))
    return ", ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load allergen database → ingredient → [canonical allergen] lookup
# ─────────────────────────────────────────────────────────────────────────────
print("Loading allergen database...")
allergen_db   = pd.read_csv(DB_PATH)
allergen_rows = allergen_db[allergen_db["group_type"] == "allergen"].copy()

ingredient_to_allergens: dict[str, set] = defaultdict(set)
for _, row in allergen_rows.iterrows():
    ingredient = str(row["ingredient"]).lower().strip()
    allergen   = canonicalise_group(str(row["group_name"]).strip())
    ingredient_to_allergens[ingredient].add(allergen)

# Longest first so multi-word ingredients match before their substrings
sorted_ingredients = sorted(ingredient_to_allergens.keys(), key=len, reverse=True)
print(f"  {len(sorted_ingredients)} allergen ingredient keywords loaded")


def detect_allergens(text: str) -> list[str]:
    """
    Return sorted, deduplicated canonical allergen names found in *text*.
    Returns empty list if none found.
    """
    text_lower = text.lower()
    found: set[str] = set()
    for ingredient in sorted_ingredients:
        if re.search(r"\b" + re.escape(ingredient) + r"\b", text_lower):
            found.update(ingredient_to_allergens[ingredient])
    return sorted(found)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load & clean restaurant-menus.csv
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nLoading Uber Eats menus from: {args.menus}")
menus = pd.read_csv(args.menus, nrows=args.limit)

menus["name"]        = menus["name"].fillna("").str.replace("&amp;", "&", regex=False)
menus["description"] = menus["description"].fillna("")
menus["category"]    = menus["category"].fillna("")

print(f"  Loaded {len(menus):,} menu items")
menus["search_text"] = menus["name"] + " " + menus["category"] + " " + menus["description"]

# ─────────────────────────────────────────────────────────────────────────────
# 3. Auto-label
# ─────────────────────────────────────────────────────────────────────────────
print("Auto-labelling items...")
menus["allergen_list"] = menus["search_text"].apply(detect_allergens)
menus["Allergens"]     = menus["allergen_list"].apply(
    lambda x: ", ".join(x) if x else None
)
menus["Prediction"]    = menus["Allergens"].apply(
    lambda x: "Contains" if pd.notna(x) else "Does not contain"
)

positives = menus[menus["Prediction"] == "Contains"]
negatives = menus[menus["Prediction"] == "Does not contain"].sample(
    min(len(positives), len(menus[menus["Prediction"] == "Does not contain"])),
    random_state=42
)
menus_labeled = pd.concat([positives, negatives]).reset_index(drop=True)

print(f"  Allergen-positive : {len(positives):,}")
print(f"  Allergen-negative : {len(negatives):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Reshape to match training schema
# ─────────────────────────────────────────────────────────────────────────────
uber_training = pd.DataFrame({
    "Food Product":    menus_labeled["name"],
    "Main Ingredient": menus_labeled["category"],
    "Sweetener":       "None",
    "Fat/Oil":         "None",
    "Seasoning":       menus_labeled["description"],
    "Allergens":       menus_labeled["Allergens"],
    "Prediction":      menus_labeled["Prediction"],
    "source":          "uber_eats",
})

# ─────────────────────────────────────────────────────────────────────────────
# 5. Load existing dataset, normalise its labels, and merge
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nLoading existing dataset: {EXISTING_PATH}")
existing = pd.read_csv(EXISTING_PATH)
existing = existing.drop_duplicates()
existing["Allergens"] = existing["Allergens"].apply(normalise_existing_label)
existing["source"]    = "original"
print(f"  {len(existing):,} rows in existing dataset")

combined = pd.concat([existing, uber_training], ignore_index=True)
combined = combined.drop_duplicates(subset=["Food Product", "Main Ingredient", "Allergens"])

print(f"\nCombined dataset: {len(combined):,} rows")
print(f"  Allergen classes : {combined['Allergens'].dropna().nunique()}")
print(f"  Label distribution:\n{combined['Prediction'].value_counts().to_string()}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Save
# ─────────────────────────────────────────────────────────────────────────────
output_path = Path(args.output)

label_counts = combined["Allergens"].value_counts()
combined = combined[combined["Allergens"].isin(label_counts[label_counts >= 5].index)]
print(f"After filtering rare labels: {len(combined):,} rows, {combined['Allergens'].nunique()} classes")

combined.to_csv(output_path, index=False)
print(f"\nSaved → {output_path}")

print("\nTop 15 allergen classes:")
print(combined["Allergens"].value_counts().head(15).to_string())
