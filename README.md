# AllerScan

AllerScan is an image-based food safety tool that helps users identify possible allergens before buying or eating food. A user can upload an image of a menu, food label, ingredient list, shelf sign, or product display, and AllerScan uses OCR and allergen detection logic to return possible allergens with confidence scores.

## Project Goal

Many people with food allergies have to make quick decisions from incomplete food information. AllerScan is designed to help by combining:

- Optical Character Recognition, or OCR, to extract food names from images
- Rule-based allergen matching using known ingredient keywords
- A multi-label machine learning model that can predict more than one allergen per food item
- CSV output for reviewing and saving allergen prediction results

This project is not intended to replace professional medical advice or official ingredient/allergen labels. It is a support tool for identifying possible risks.

## Features

- Upload or process a menu/food image
- Extract text from the image using OCR
- Detect food items from extracted text
- Predict possible allergens for each food item
- Supports multi-label allergen prediction, meaning one food item can have multiple allergens
- Uses rule-based matching first for high-confidence ingredient matches
- Falls back to a machine learning model when direct matches are not found
- Exports prediction results to a CSV file

## Repository Structure

```bash
AllerScan/
├── data/
│   ├── sample/
│   │   ├── sample_allergen_database.csv
│   │   └── sample_menu.jpg
│   ├── food_ingredients_and_allergens.csv
│   └── training_data.csv
│
├── notebooks/
│   ├── OCR.ipynb
│   ├── foodclassification.ipynb
│   ├── mlmodel.ipynb
│   └── rule_based_matching.ipynb
│
├── src/
│   ├── build_training_data.py
│   ├── predict_allergens.py
│   └── train_allergen_model.py
│
├── models/
│   └── .gitkeep
│
├── outputs/
│   └── .gitkeep
│
├── README.md
├── DATA.md
├── .gitignore
└── UI_Design.py