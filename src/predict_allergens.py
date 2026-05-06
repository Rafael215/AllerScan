import re
import joblib
import pandas as pd
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "allergen_model.pkl"
DATABASE_PATH = ROOT / "allergen_database.csv"

model = joblib.load(MODEL_PATH)
allergen_db = pd.read_csv(DATABASE_PATH)

CANONICAL = {
    "milk": "Dairy",
    "egg": "Eggs",
    "peanut": "Peanuts",
    "tree_nut": "Tree Nuts",
    "soy": "Soybeans",
    "wheat": "Wheat",
    "gluten": "Wheat",
    "fish": "Fish",
    "shellfish": "Shellfish",
    "sesame": "Sesame",
    "mustard": "Mustard",
    "celery": "Celery",
    "corn": "Corn",
    "coconut": "Coconut",
    "garlic": "Garlic",
    "onion": "Onion",
    "legume": "Legumes",
}

DISH_INGREDIENT_HINTS = {
    "burger": "bun bread wheat beef cheese",
    "chicken": "chicken breaded batter wheat egg",
    "fried": "fried batter flour wheat egg",
    "fish": "fish",
    "chips": "potato oil",
    "spaghetti": "pasta wheat tomato",
    "meatball": "beef pork egg breadcrumbs wheat",
    "hotdog": "beef pork bun wheat",
    "sandwich": "bread wheat",
    "macaroni": "pasta wheat cheese dairy",
    "soup": "broth celery onion garlic",
    "chili": "beef beans onion garlic",
    "fries": "potato oil",
    "calamari": "squid shellfish batter wheat egg",
    "taco": "tortilla corn wheat beef cheese",
    "cheese": "cheese dairy",
    "cream": "cream dairy",
    "juice": "fruit",
    "lemonade": "lemon sugar",
    "soda": "sugar carbonated water",
    "water": "water",
}


def canonicalize(label):
    label = str(label).strip().lower()
    return CANONICAL.get(label, label.title())


def whole_word_match(needle, haystack):
    return bool(re.search(r"\b" + re.escape(needle) + r"\b", haystack))


def build_ingredient_lookup():
    allergen_rows = allergen_db[allergen_db["group_type"] == "allergen"]

    lookup = defaultdict(set)

    for _, row in allergen_rows.iterrows():
        ingredient = str(row["ingredient"]).lower().strip()
        allergen = canonicalize(row["group_name"])

        if ingredient:
            lookup[ingredient].add(allergen)

    return lookup


INGREDIENT_LOOKUP = build_ingredient_lookup()
SORTED_INGREDIENTS = sorted(INGREDIENT_LOOKUP.keys(), key=len, reverse=True)


def enrich_food_item(food_item):
    item_lower = food_item.lower()
    hints = []

    for keyword, extra_words in DISH_INGREDIENT_HINTS.items():
        if whole_word_match(keyword, item_lower):
            hints.append(extra_words)

    if hints:
        return food_item + " " + " ".join(hints)

    return food_item


def rule_based_allergens(food_item):
    """
    Detect all allergens that directly or indirectly match the food item.
    Uses the original food name plus dish hints.
    """
    search_text = enrich_food_item(food_item).lower()

    found = {}

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

    final_matches = []

    for allergen, info in found.items():
        final_matches.append({
            "allergen": allergen,
            "matched_ingredients": sorted(info["matched_ingredients"]),
            "confidence": info["confidence"],
        })

    return sorted(final_matches, key=lambda x: x["allergen"])


def split_ml_labels(raw_label):
    if pd.isna(raw_label):
        return []

    return sorted(set(label.strip() for label in str(raw_label).split(",") if label.strip()))


def ml_allergens(food_item):
    enriched = enrich_food_item(food_item)

    prediction = model.predict([enriched])[0]
    probabilities = model.predict_proba([enriched])[0]
    confidence = float(max(probabilities))

    labels = split_ml_labels(prediction)

    return {
        "allergens": labels,
        "confidence": round(confidence, 3),
        "debug_input": enriched,
    }


def predict_allergens(food_items, ml_threshold=0.50):
    results = []

    for item in food_items:
        rule_matches = rule_based_allergens(item)

        if rule_matches:
            allergens = [m["allergen"] for m in rule_matches]
            matched_ingredients = sorted(set(
                ingredient
                for match in rule_matches
                for ingredient in match["matched_ingredients"]
            ))

            confidence = max(m["confidence"] for m in rule_matches)

            results.append({
                "food_item": item,
                "predicted_allergens": allergens,
                "matched_ingredients": matched_ingredients,
                "method": "rule-based + ingredient hints",
                "confidence": round(confidence, 3),
            })

        else:
            ml = ml_allergens(item)

            if ml["confidence"] < ml_threshold:
                allergens = ["uncertain"]
            else:
                allergens = ml["allergens"]

            results.append({
                "food_item": item,
                "predicted_allergens": allergens,
                "matched_ingredients": ["ML prediction"],
                "method": "ML model",
                "confidence": ml["confidence"],
                "debug_input": ml["debug_input"],
            })

    return results


if __name__ == "__main__":
    food_items = [
        "Mushroom Burger",
        "Crispy Fried Chicken",
        "Fish & Chips",
        "Spaghetti & Meatballs",
        "Hotdog Sandwich",
        "Macaroni Soup",
        "Tex Mex Chili",
        "French Fries",
        "Calamari",
        "Beef Taco",
        "Purified Water",
        "Sparkling Water",
        "Soda In A Bottle",
        "Orange Juice",
        "Fresh Lemonade",
    ]

    results = predict_allergens(food_items)

    for r in results:
        print(r["food_item"])
        print("  Allergens:", ", ".join(r["predicted_allergens"]))
        print("  Matched ingredients:", ", ".join(r["matched_ingredients"]))
        print("  Method:", r["method"])
        print("  Confidence:", f"{r['confidence']:.1%}")
        print()