"""
predict_allergens.py  –  Predict allergens for a list of food items

Multi-label aware: loads the MultiLabelBinarizer saved by train_allergen_model.py
so each allergen is decoded independently from the model's binary outputs.

Changes vs. original:
  1. Loads allergen_binarizer.pkl alongside allergen_model.pkl.
  2. ml_allergens() now returns per-allergen probabilities, giving a much
     more meaningful confidence score (average prob across predicted classes
     rather than max of a flat multi-class softmax).
  3. Items with no allergen hits (water, juice, etc.) now get a confident
     "None" result instead of "uncertain", because the "None" class is
     part of the binarizer vocabulary.
  4. ml_threshold applied per allergen, not to the whole prediction, so
     borderline allergens are filtered individually.
  5. Rule-based layer unchanged — still runs first for direct keyword hits.
"""

import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH    = ROOT / "models" / "allergen_model.pkl"
MLB_PATH      = ROOT / "models" / "allergen_binarizer.pkl"
DATABASE_PATH = ROOT / "data" / "sample" / "sample_allergen_database.csv"

# Load model + binarizer
model = joblib.load(MODEL_PATH)
mlb   = joblib.load(MLB_PATH)

allergen_db = pd.read_csv(DATABASE_PATH)

# ── Canonical allergen name map ───────────────────────────────────────────────
CANONICAL = {
    "milk":      "Dairy",
    "egg":       "Eggs",
    "peanut":    "Peanuts",
    "tree_nut":  "Tree Nuts",
    "soy":       "Soybeans",
    "wheat":     "Wheat",
    "gluten":    "Wheat",
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

# ── Dish-level ingredient hints ───────────────────────────────────────────────
DISH_INGREDIENT_HINTS = {
    # existing
    "burger":       "bun bread wheat beef cheese dairy",
    "fried":        "fried batter flour wheat egg",
    "fish":         "fish",
    "chips":        "potato oil",
    "spaghetti":    "pasta wheat tomato egg",
    "meatball":     "beef pork egg breadcrumbs wheat",
    "hotdog":       "beef pork bun wheat",
    "sandwich":     "bread wheat",
    "macaroni":     "pasta wheat cheese dairy",
    "soup":         "broth celery onion garlic",
    "chili":        "beef beans onion garlic",
    "fries":        "potato oil",
    "calamari":     "squid shellfish batter wheat egg",
    "taco":         "tortilla corn wheat beef cheese dairy",
    "cheese":       "cheese dairy milk",
    "cream":        "cream dairy milk",
    "juice":        "fruit",
    "lemonade":     "lemon sugar water",
    "soda":         "sugar carbonated water",
    "water":        "water",
    "sparkling":    "water carbonated",
    # fixed — plain grilled chicken should NOT get batter/breaded hints
    "grilled chicken": "chicken",
    "chicken":      "chicken egg wheat batter breaded",  # fallback for non-grilled
    # new dish hints
    "lasagna":      "pasta wheat beef egg cheese dairy milk",
    "pizza":        "dough wheat cheese dairy milk tomato",
    "gnocchi":      "potato wheat egg flour",
    "pad thai":     "rice noodle egg peanut shrimp soy",
    "satay":        "peanut soy sauce wheat",
    "slider":       "bun bread wheat",
    "pulled pork":  "pork wheat bread bun",
    "spring roll":  "wheat wrapper egg soy",
    "caesar":       "anchovy fish egg dairy parmesan wheat crouton",
    "brownie":      "wheat egg butter dairy chocolate",
    "croissant":    "wheat butter dairy egg milk",
    "omelette":     "egg dairy milk cheese",
    "bisque":       "cream dairy milk shellfish wheat",
    "chowder":      "cream dairy milk shellfish potato wheat",
    "wrap":         "tortilla wheat",
    "latte":        "milk dairy espresso",
    "curry":        "coconut milk dairy",
    "miso":         "soy miso tofu",
    "edamame":      "soy soybean",
    "waffle":       "wheat egg milk dairy butter",
    "pancake":      "wheat egg milk dairy butter",
    "french toast": "wheat egg milk dairy",
    "cheesecake":   "wheat egg dairy milk cream cheese",
    "tiramisu":     "egg dairy milk wheat coffee",
    "risotto":      "dairy milk butter parmesan",
    "ravioli":      "wheat egg pasta dairy cheese",
    "crab cake":    "shellfish crab egg wheat breadcrumb",
    "fish cake":    "fish egg wheat",
    "tempura":      "wheat flour egg batter shellfish",
    "dumpling":     "wheat egg pork soy",
    "ramen":        "wheat noodle egg soy pork",
    "udon":         "wheat noodle egg soy",
    "sushi":        "fish shellfish soy rice",
    "katsu":        "wheat egg breadcrumb pork chicken",
    "gyoza":        "wheat egg pork soy",
    "pesto":        "tree nuts pine nuts dairy parmesan wheat",
    "hummus":       "legume chickpea sesame tahini",
    "falafel":      "legume chickpea wheat",
    "shawarma":     "wheat bread egg dairy",
    "naan":         "wheat dairy milk egg",
    "pretzel":      "wheat egg",
    "bagel":        "wheat egg",
    "muffin":       "wheat egg dairy milk butter",
    "scone":        "wheat egg dairy milk butter",
    "cookie":       "wheat egg butter dairy",
    "cake":         "wheat egg butter dairy milk",
    "pudding":      "egg dairy milk wheat",
    "ice cream":    "dairy milk egg cream",
    "gelato":       "dairy milk egg cream",
    "mousse":       "egg dairy cream chocolate",
}

# Known clean items: no allergen expected
CLEAN_ITEMS = {"water", "sparkling water", "purified water", "mineral water"}


def canonicalize(label: str) -> str:
    label = str(label).strip().lower()
    return CANONICAL.get(label, label.title())


def whole_word_match(needle: str, haystack: str) -> bool:
    return bool(re.search(r"\b" + re.escape(needle) + r"\b", haystack))


# ── Build ingredient → allergen lookup ───────────────────────────────────────
def build_ingredient_lookup():
    allergen_rows = allergen_db[allergen_db["group_type"] == "allergen"]
    lookup: dict[str, set] = defaultdict(set)
    for _, row in allergen_rows.iterrows():
        ingredient = str(row["ingredient"]).lower().strip()
        allergen   = canonicalize(row["group_name"])
        if ingredient:
            lookup[ingredient].add(allergen)
    return lookup


INGREDIENT_LOOKUP   = build_ingredient_lookup()
SORTED_INGREDIENTS  = sorted(INGREDIENT_LOOKUP.keys(), key=len, reverse=True)


def enrich_food_item(food_item: str) -> str:
    """
    Append dish-level ingredient hints to the food item text.
    Matches longest keywords first so "grilled chicken" takes priority over
    the generic "chicken" hint, preventing false batter/breaded signals.
    """
    item_lower = food_item.lower()
    hints = []
    matched_spans: list[tuple[int, int]] = []

    for keyword in sorted(DISH_INGREDIENT_HINTS.keys(), key=len, reverse=True):
        m = re.search(r"\b" + re.escape(keyword) + r"\b", item_lower)
        if not m:
            continue
        start, end = m.start(), m.end()
        if any(s <= start and end <= e for s, e in matched_spans):
            continue
        matched_spans.append((start, end))
        hints.append(DISH_INGREDIENT_HINTS[keyword])

    return (food_item + " " + " ".join(hints)).strip() if hints else food_item



# ── Rule-based detection ──────────────────────────────────────────────────────
def rule_based_allergens(food_item: str) -> list[dict]:
    search_text = enrich_food_item(food_item).lower()
    found: dict[str, dict] = {}

    for ingredient in SORTED_INGREDIENTS:
        if whole_word_match(ingredient, search_text):
            for allergen in INGREDIENT_LOOKUP[ingredient]:
                if allergen not in found:
                    found[allergen] = {
                        "allergen": allergen,
                        "matched_ingredients": set(),
                        "confidence": 0.95,
                    }
                found[allergen]["matched_ingredients"].add(ingredient)

    return sorted(
        [
            {**info, "matched_ingredients": sorted(info["matched_ingredients"])}
            for info in found.values()
        ],
        key=lambda x: x["allergen"],
    )


# ── ML-based detection (multi-label aware) ───────────────────────────────────
# ── Per-allergen thresholds ───────────────────────────────────────────────────
# Lower = more sensitive = more false positives but fewer dangerous misses.
# Eggs and Dairy are very common hidden allergens so they get extra sensitivity.
ALLERGEN_THRESHOLDS: dict[str, float] = {
    "Eggs":  0.03,   # 3%  — very sensitive, eggs hide in almost everything
    "Dairy": 0.04,   # 4%  — very sensitive, dairy hides in sauces/dressings
    # everything else uses the default below
}
DEFAULT_THRESHOLD = 0.12


def ml_allergens(food_item: str, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Use the multi-label model to predict allergens.

    Returns per-allergen probabilities; only allergens whose probability
    exceeds `threshold` are included in the result.

    Confidence reported is the mean probability across all predicted
    allergens (or the highest "None" probability if nothing is detected).
    """
    enriched = enrich_food_item(food_item)

    # predict_proba returns shape (n_samples, n_classes) for OvR
    proba_matrix = model.predict_proba([enriched])   # shape (1, n_classes)
    proba        = proba_matrix[0]                    # shape (n_classes,)

    classes = mlb.classes_  # e.g. ["Celery", "Dairy", "Eggs", "None", ...]

    # Build per-allergen result above threshold
    detected = []
    for cls, prob in zip(classes, proba):
        if cls == "None":
            continue
        cutoff = ALLERGEN_THRESHOLDS.get(cls, threshold)
        if prob >= cutoff:
            detected.append({"allergen": cls, "probability": round(float(prob), 3)})

    # Confidence: mean of detected probs; fall back to "None" class prob
    if detected:
        confidence = float(np.mean([d["probability"] for d in detected]))
        allergens  = [d["allergen"] for d in detected]
    else:
        none_idx   = list(classes).index("None") if "None" in classes else -1
        confidence = float(proba[none_idx]) if none_idx >= 0 else 1.0
        allergens  = []  # genuinely clean

    return {
        "allergens":   sorted(allergens),
        "confidence":  round(confidence, 3),
        "debug_input": enriched,
        "per_allergen_proba": {cls: round(float(p), 3) for cls, p in zip(classes, proba)},
    }


# ── Main predict function ─────────────────────────────────────────────────────
def predict_allergens(food_items: list[str], ml_threshold: float = 0.12) -> list[dict]:
    results = []

    for item in food_items:
        # Fast-path: known clean items
        if item.lower().strip() in CLEAN_ITEMS:
            results.append({
                "food_item":           item,
                "predicted_allergens": [],
                "matched_ingredients": [],
                "method":              "known clean item",
                "confidence":          1.0,
            })
            continue

        # Rule-based first (highest precision)
        rule_matches = rule_based_allergens(item)

        if rule_matches:
            results.append({
                "food_item":           item,
                "predicted_allergens": [m["allergen"] for m in rule_matches],
                "matched_ingredients": sorted(set(
                    ing for m in rule_matches for ing in m["matched_ingredients"]
                )),
                "method":              "rule-based + ingredient hints",
                "confidence":          round(max(m["confidence"] for m in rule_matches), 3),
            })
        else:
            # Fall back to multi-label ML model
            ml = ml_allergens(item, threshold=ml_threshold)

            results.append({
                "food_item":           item,
                "predicted_allergens": ml["allergens"] if ml["allergens"] else ["None"],
                "matched_ingredients": ["ML prediction"],
                "method":              "ML model (multi-label)",
                "confidence":          ml["confidence"],
                "debug_input":         ml.get("debug_input"),
                "per_allergen_proba":  ml.get("per_allergen_proba"),
            })

    return results


# ── CLI demo ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    food_items = [
    # tricky dishes with hidden allergens
    "Carbonara",
    "Bolognese",
    "Katsu Curry",
    "Laksa",
    "Bibimbap",
    "Banh Mi",
    "Pho",
    "Shakshuka",
    "Bouillabaisse",
    "Moussaka",
    "Stroganoff",
    "Kedgeree",
    "Khao Pad",
    "Jollof Rice",
    "Rendang",

    # desserts with no obvious keywords
    "Tiramisu",
    "Profiteroles",
    "Baklava",
    "Cannoli",
    "Panna Cotta",
    "Churros",
    "Crepes",
    "Madeleines",
    "Financiers",
    "Kouign Amann",

    # drinks / clean items
    "Espresso",
    "Matcha",
    "Kombucha",
    "Cold Brew",
    "Americano",
    "Breakfast Plate"
]

    results = predict_allergens(food_items)

    for r in results:
        allergen_str = ", ".join(r["predicted_allergens"]) if r["predicted_allergens"] else "None"
        print(r["food_item"])
        print(f"  Allergens          : {allergen_str}")
        print(f"  Matched ingredients: {', '.join(r['matched_ingredients'])}")
        print(f"  Method             : {r['method']}")
        print(f"  Confidence         : {r['confidence']:.1%}")

        # Show per-allergen breakdown for ML predictions
        if r.get("per_allergen_proba"):
            top = sorted(r["per_allergen_proba"].items(), key=lambda x: -x[1])[:5]
            print(f"  Top allergen probs : { {k: f'{v:.1%}' for k, v in top} }")
        print()