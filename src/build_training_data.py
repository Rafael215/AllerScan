"""
build_training_data.py  –  Convert restaurant-menus.csv into labeled training data

Pipeline:
  1. Load restaurant-menus.csv (Uber Eats dataset)
  2. Auto-label each item using allergen_database.csv keyword matching
  3. Normalise all allergen labels to a single canonical vocabulary
  4. Merge with existing food_ingredients_and_allergens.csv labels
  5. Output a unified training CSV ready for train_allergen_model.py

Changes vs. original:
  - Negative examples (no allergen) are labelled "None" instead of NULL,
    so the multi-label model sees a real "None" class during training.
  - Negative sample count capped at 2× positives (was 1×) for a better
    balance without drowning allergen signal.
  - rare-label threshold raised to 5 (unchanged) but applied after merge.

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

# ── Dish-level ingredient hints (must stay in sync with predict_allergens.py) ─
# Applied to the name before keyword matching so dishes with no description
# (e.g. "Carbonara" with null description) still get correctly labeled.
DISH_INGREDIENT_HINTS: dict[str, str] = {
    "grilled chicken": "chicken",
    "chicken":         "chicken egg wheat batter breaded",
    "burger":          "bun bread wheat beef cheese dairy",
    "fried":           "fried batter flour wheat egg",
    "fish":            "fish",
    "chips":           "potato oil",
    "spaghetti":       "pasta wheat tomato egg",
    "meatball":        "beef pork egg breadcrumbs wheat",
    "hotdog":          "beef pork bun wheat",
    "sandwich":        "bread wheat",
    "macaroni":        "pasta wheat cheese dairy",
    "soup":            "broth celery onion garlic",
    "chili":           "beef beans onion garlic",
    "fries":           "potato oil",
    "calamari":        "squid shellfish batter wheat egg",
    "taco":            "tortilla corn wheat beef cheese dairy",
    "cheese":          "cheese dairy milk",
    "cream":           "cream dairy milk",
    "juice":           "fruit",
    "lemonade":        "lemon sugar water",
    "soda":            "sugar carbonated water",
    "water":           "water",
    "sparkling":       "water carbonated",
    "lasagna":         "pasta wheat beef egg cheese dairy milk",
    "pizza":           "dough wheat cheese dairy milk tomato",
    "gnocchi":         "potato wheat egg flour",
    "pad thai":        "rice noodle egg peanut shrimp soy",
    "satay":           "peanut soy sauce wheat",
    "slider":          "bun bread wheat",
    "pulled pork":     "pork wheat bread bun",
    "spring roll":     "wheat wrapper egg soy",
    "caesar":          "anchovy fish egg dairy parmesan wheat crouton",
    "brownie":         "wheat egg butter dairy chocolate",
    "croissant":       "wheat butter dairy egg milk",
    "omelette":        "egg dairy milk cheese",
    "bisque":          "cream dairy milk shellfish wheat",
    "chowder":         "cream dairy milk shellfish potato wheat",
    "wrap":            "tortilla wheat",
    "latte":           "milk dairy espresso",
    "curry":           "coconut milk dairy",
    "miso":            "soy miso tofu",
    "edamame":         "soy soybean",
    "carbonara":       "pasta wheat egg dairy bacon",
    "bolognese":       "pasta wheat beef egg dairy",
    "laksa":           "coconut milk shellfish egg noodle wheat",
    "bibimbap":        "egg soy sesame rice beef",
    "banh mi":         "bread wheat egg pork",
    "pho":             "noodle wheat beef broth",
    "shakshuka":       "egg tomato onion garlic",
    "bouillabaisse":   "fish shellfish onion garlic",
    "moussaka":        "egg dairy milk wheat beef",
    "stroganoff":      "beef dairy cream egg wheat",
    "kedgeree":        "fish egg dairy milk rice",
    "khao pad":        "egg soy rice",
    "rendang":         "coconut milk beef",
    "tiramisu":        "egg dairy milk wheat coffee",
    "profiteroles":    "wheat egg dairy milk butter",
    "baklava":         "tree nuts wheat butter dairy honey",
    "cannoli":         "wheat egg dairy milk ricotta",
    "panna cotta":     "dairy milk cream egg",
    "churros":         "wheat egg oil",
    "crepes":          "wheat egg dairy milk butter",
    "madeleines":      "wheat egg dairy milk butter",
    "financiers":      "wheat egg dairy butter tree nuts almond",
    "kouign amann":    "wheat egg dairy butter milk",
    "waffle":          "wheat egg milk dairy butter",
    "pancake":         "wheat egg milk dairy butter",
    "french toast":    "wheat egg milk dairy",
    "cheesecake":      "wheat egg dairy milk cream cheese",
    "risotto":         "dairy milk butter parmesan",
    "ravioli":         "wheat egg pasta dairy cheese",
    "crab cake":       "shellfish crab egg wheat breadcrumb",
    "tempura":         "wheat flour egg batter shellfish",
    "dumpling":        "wheat egg pork soy",
    "ramen":           "wheat noodle egg soy pork",
    "udon":            "wheat noodle egg soy",
    "sushi":           "fish shellfish soy rice",
    "katsu":           "wheat egg breadcrumb pork chicken",
    "gyoza":           "wheat egg pork soy",
    "pesto":           "tree nuts pine nuts dairy parmesan wheat",
    "hummus":          "legume chickpea sesame tahini",
    "falafel":         "legume chickpea wheat",
    "shawarma":        "wheat bread egg dairy",
    "naan":            "wheat dairy milk egg",
    "pretzel":         "wheat egg",
    "bagel":           "wheat egg",
    "muffin":          "wheat egg dairy milk butter",
    "scone":           "wheat egg dairy milk butter",
    "cookie":          "wheat egg butter dairy",
    "cake":            "wheat egg butter dairy milk",
    "pudding":         "egg dairy milk wheat",
    "ice cream":       "dairy milk egg cream",
    "gelato":          "dairy milk egg cream",
    "mousse":          "egg dairy cream chocolate",
}


def enrich_text(text: str) -> str:
    """
    Append dish-level ingredient hints to text before keyword matching.
    Longest keywords matched first so 'grilled chicken' beats 'chicken'.
    """
    text_lower = text.lower()
    hints = []
    matched_spans: list[tuple[int, int]] = []

    for keyword in sorted(DISH_INGREDIENT_HINTS.keys(), key=len, reverse=True):
        m = re.search(r"\b" + re.escape(keyword) + r"\b", text_lower)
        if not m:
            continue
        start, end = m.start(), m.end()
        if any(s <= start and end <= e for s, e in matched_spans):
            continue
        matched_spans.append((start, end))
        hints.append(DISH_INGREDIENT_HINTS[keyword])

    return (text + " " + " ".join(hints)).strip() if hints else text

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
MENUS_PATH    = ROOT / "data" / "restaurant-menus.csv"
DB_PATH       = ROOT / "data" / "sample" / "sample_allergen_database.csv"
EXISTING_PATH = ROOT / "data" / "food_ingredients_and_allergens.csv"
OUTPUT_PATH   = ROOT / "data" / "training_data.csv"

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--menus",  default=str(MENUS_PATH), help="Path to restaurant-menus.csv")
parser.add_argument("--limit",  type=int, default=None,  help="Max rows to load from menus CSV")
parser.add_argument("--output", default=str(OUTPUT_PATH),help="Output training CSV path")
args = parser.parse_args()

# ── Canonical allergen name map ───────────────────────────────────────────────
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
}


def canonicalise_group(raw: str) -> str:
    return CANONICAL.get(raw, raw.title())


def normalise_existing_label(label: str) -> str:
    """Fix known bad labels and re-sort components. NULL → 'None'."""
    if pd.isna(label):
        return "None"   # CHANGED: explicit "None" class instead of NaN
    label = EXISTING_LABEL_FIX.get(label, label)
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

sorted_ingredients = sorted(ingredient_to_allergens.keys(), key=len, reverse=True)
print(f"  {len(sorted_ingredients)} allergen ingredient keywords loaded")


def detect_allergens(text: str) -> list[str]:
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

# KEY FIX: enrich with dish hints BEFORE labeling so dishes with null/vague
# descriptions (e.g. "Carbonara") still get correctly labeled from their name alone.
# This makes training input identical to what predict_allergens.py uses at runtime.
print("Enriching search text with dish hints...")
menus["search_text"] = menus["search_text"].apply(enrich_text)
enriched_count = (menus["search_text"] != (menus["name"] + " " + menus["category"] + " " + menus["description"])).sum()
print(f"  {enriched_count:,} items enriched with dish hints")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Auto-label
# ─────────────────────────────────────────────────────────────────────────────
print("Auto-labelling items...")
menus["allergen_list"] = menus["search_text"].apply(detect_allergens)
menus["Allergens"]     = menus["allergen_list"].apply(
    lambda x: ", ".join(x) if x else "None"   # CHANGED: "None" not NULL
)
menus["Prediction"]    = menus["Allergens"].apply(
    lambda x: "Contains" if x != "None" else "Does not contain"
)

positives = menus[menus["Prediction"] == "Contains"]
# CHANGED: allow up to 2× positives for negatives (better balance)
max_neg = min(len(positives) * 2, len(menus[menus["Prediction"] == "Does not contain"]))
negatives = menus[menus["Prediction"] == "Does not contain"].sample(
    max_neg, random_state=42
)
menus_labeled = pd.concat([positives, negatives]).reset_index(drop=True)

print(f"  Allergen-positive : {len(positives):,}")
print(f"  Allergen-negative : {len(negatives):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Reshape to match training schema
# ─────────────────────────────────────────────────────────────────────────────
uber_training = pd.DataFrame({
    "Food Product":    menus_labeled["name"].apply(enrich_text),  # enriched so model trains on same input as prediction
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
print(f"  Allergen classes : {combined['Allergens'].nunique()}")
print(f"  Label distribution:\n{combined['Prediction'].value_counts().to_string()}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Filter rare labels and save
# ─────────────────────────────────────────────────────────────────────────────
output_path = Path(args.output)

label_counts = combined["Allergens"].value_counts()
# Always keep "None" regardless of count
keep_mask = combined["Allergens"].isin(
    label_counts[label_counts >= 5].index
) | (combined["Allergens"] == "None")
combined = combined[keep_mask]

print(f"After filtering rare labels: {len(combined):,} rows, {combined['Allergens'].nunique()} classes")

combined.to_csv(output_path, index=False)
print(f"\nSaved → {output_path}")

print("\nTop 15 allergen classes:")
print(combined["Allergens"].value_counts().head(15).to_string())