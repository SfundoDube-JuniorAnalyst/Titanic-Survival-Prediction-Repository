"""
predict.py — Shared preprocessing + prediction logic for the Titanic Survival model.
"""
import os
import numpy as np
import pandas as pd
import joblib

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

model = joblib.load(os.path.join(MODEL_DIR, "optimized_titanic_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
selected_features = joblib.load(os.path.join(MODEL_DIR, "selected_features.pkl"))

# Exact column order the scaler was fit on during training
SCALER_NUMERIC_COLS = [
    "Pclass", "Age", "SibSp", "Parch", "Fare_log",
    "FamilySize", "FarePerPerson_log", "TicketFreq"
]

VALID_TITLES = ["Mr", "Mrs", "Miss", "Master", "Officer", "Royalty"]
VALID_SEX = ["male", "female"]
VALID_PCLASS = [1, 2, 3]


def engineer_features(passenger: dict) -> dict:
    """Takes raw passenger input and computes every engineered feature used in training."""
    pclass = passenger["Pclass"]
    sex = passenger["Sex"]
    age = passenger["Age"]
    sibsp = passenger["SibSp"]
    parch = passenger["Parch"]
    fare = passenger["Fare"]
    title = passenger["Title"]
    cabin_known = passenger["CabinKnown"]
    ticket_freq = passenger.get("TicketFreq", 1)

    family_size = sibsp + parch + 1
    fare_log = np.log1p(fare)
    fare_per_person = fare / family_size
    fare_per_person_log = np.log1p(fare_per_person)

    sex_enc = 1 if sex == "male" else 0  # female=0, male=1
    title_mr = 1 if title == "Mr" else 0
    title_mrs = 1 if title == "Mrs" else 0

    pclass_sex = f"{pclass}_{sex}"
    pclass_sex_3_male = 1 if pclass_sex == "3_male" else 0
    pclass_sex_2_female = 1 if pclass_sex == "2_female" else 0

    deck_unknown = 0 if cabin_known else 1

    return {
        "Pclass": pclass, "Age": age, "SibSp": sibsp, "Parch": parch,
        "Fare_log": fare_log, "FamilySize": family_size,
        "FarePerPerson_log": fare_per_person_log, "TicketFreq": ticket_freq,
        "Sex_enc": sex_enc, "Title_Mr": title_mr, "Title_Mrs": title_mrs,
        "Pclass_Sex_3_male": pclass_sex_3_male,
        "Pclass_Sex_2_female": pclass_sex_2_female,
        "Deck_Unknown": deck_unknown,
    }


def build_feature_vector(engineered: dict) -> pd.DataFrame:
    """Scales numeric features matching training scaler and outputs model feature vector."""
    numeric_df = pd.DataFrame([{c: engineered[c] for c in SCALER_NUMERIC_COLS}])
    scaled_values = scaler.transform(numeric_df)[0]
    scaled_dict = dict(zip(SCALER_NUMERIC_COLS, scaled_values))

    full_row = {**engineered, **scaled_dict}

    return pd.DataFrame([{f: full_row[f] for f in selected_features}])


def validate_input(passenger: dict):
    errors = []
    if passenger.get("Pclass") not in VALID_PCLASS:
        errors.append(f"Pclass must be one of {VALID_PCLASS}")
    if passenger.get("Sex") not in VALID_SEX:
        errors.append(f"Sex must be one of {VALID_SEX}")
    if passenger.get("Title") not in VALID_TITLES:
        errors.append(f"Title must be one of {VALID_TITLES}")
    # Fixed: allow 0 <= Age <= 100 for infant passengers
    if not (0 <= passenger.get("Age", -1) <= 100):
        errors.append("Age must be between 0 and 100")
    if passenger.get("Fare", -1) < 0:
        errors.append("Fare must be non-negative")
    if passenger.get("SibSp", -1) < 0:
        errors.append("SibSp must be non-negative")
    if passenger.get("Parch", -1) < 0:
        errors.append("Parch must be non-negative")
    return errors


def predict_survival(passenger: dict) -> dict:
    errors = validate_input(passenger)
    if errors:
        raise ValueError("; ".join(errors))

    engineered = engineer_features(passenger)
    X = build_feature_vector(engineered)

    pred_class = int(model.predict(X)[0])
    pred_proba = float(model.predict_proba(X)[0][1])

    return {
        "prediction": pred_class,
        "prediction_label": "Survived" if pred_class == 1 else "Did not survive",
        "survival_probability": round(pred_proba, 4)
    }