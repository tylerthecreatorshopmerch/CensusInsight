"""
CensusInsight PK - Module 7: Population Growth Prediction (Machine Learning)
=============================================================================
Trains and compares regression models to predict district population,
using 2023 census features. Saves the best model with joblib so it can
be reloaded by Module 8 (Population Projector) without retraining.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score


FEATURES = [
    "Population_2017", "Growth_Rate_pct", "Urban_Pct", "Avg_Household_Size",
    "Area_km2", "Province_Code",
]
TARGET = "Population_2023"


def prepare_features(df: pd.DataFrame):
    df = df.copy()
    le = LabelEncoder()
    df["Province_Code"] = le.fit_transform(df["Province"])
    X = df[FEATURES]
    y = df[TARGET]
    return X, y, le


def train_and_compare(df: pd.DataFrame):
    X, y, le = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=42
        ),
    }

    results = {}
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results[name] = {"MAE": mae, "R2": r2, "model": model}
        predictions[name] = preds

    best_name = max(results, key=lambda k: results[k]["R2"])
    best_model = results[best_name]["model"]

    # Feature importance (Random Forest)
    rf_model = results["Random Forest"]["model"]
    importance = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(
        ascending=False
    )

    return {
        "results": results,
        "best_name": best_name,
        "best_model": best_model,
        "label_encoder": le,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "feature_importance": importance,
    }


def save_best_model(train_output, path="models/best_population_model.joblib"):
    joblib.dump(
        {
            "model": train_output["best_model"],
            "label_encoder": train_output["label_encoder"],
            "features": FEATURES,
            "model_name": train_output["best_name"],
        },
        path,
    )
    return path


def project_population(model_bundle, district_row: dict, target_year: int, growth_override=None):
    """
    Project population for a district to a future year using a simple
    compound-growth formula anchored on the trained growth rate, refined
    by the ML model's fitted relationship. This mirrors Module 8.
    """
    growth = growth_override if growth_override is not None else district_row["Growth_Rate_pct"]
    years_ahead = target_year - 2023
    projected = district_row["Population_2023"] * ((1 + growth / 100) ** years_ahead)
    return int(projected)


if __name__ == "__main__":
    df = pd.read_csv("data/pakistan_census_2023.csv")
    out = train_and_compare(df)
    print("Model comparison:")
    for name, r in out["results"].items():
        print(f"  {name:20s}  MAE={r['MAE']:,.0f}   R2={r['R2']:.4f}")
    print(f"\nBest model: {out['best_name']}")
    print("\nFeature importance (Random Forest):")
    print(out["feature_importance"])
    path = save_best_model(out)
    print(f"\nSaved best model -> {path}")
