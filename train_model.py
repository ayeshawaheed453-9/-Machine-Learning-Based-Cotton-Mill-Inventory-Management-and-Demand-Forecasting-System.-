"""Train and compare regression models for cotton-mill demand forecasting.

Run: python train_model.py
The script creates a clearly labeled synthetic dataset when the CSV is missing.
"""
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cotton_inventory.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
RANDOM_STATE = 42
PRODUCTS = ["Raw Cotton", "Cotton Yarn", "Cotton Fabric", "Cotton Thread", "Cotton Waste"]


def create_synthetic_dataset(n_rows=300):
    """Create realistic-looking, deterministic monthly mill observations."""
    rng = np.random.default_rng(RANDOM_STATE)
    dates = pd.date_range("2021-01-01", periods=36, freq="MS")
    rows = []
    base_demand = {
        "Raw Cotton": 420, "Cotton Yarn": 330, "Cotton Fabric": 280,
        "Cotton Thread": 190, "Cotton Waste": 90,
    }
    for i in range(n_rows):
        product = PRODUCTS[i % len(PRODUCTS)]
        date = dates[i % len(dates)]
        season = 1 + 0.12 * np.sin(2 * np.pi * date.month / 12)
        previous = max(20, base_demand[product] * season + rng.normal(0, 25))
        sales = max(10, previous * rng.uniform(0.82, 1.08) + rng.normal(0, 12))
        production = max(10, base_demand[product] * rng.uniform(0.75, 1.12) + rng.normal(0, 18))
        purchase = max(0, base_demand[product] * rng.uniform(0.05, 0.45) + rng.normal(0, 12))
        stock = max(10, base_demand[product] * rng.uniform(0.45, 1.55) + production * 0.10 + rng.normal(0, 25))
        future = max(5, 0.45 * previous + 0.28 * sales + 0.10 * production + 0.05 * purchase - 0.04 * stock + rng.normal(0, 16))
        rows.append([date.date().isoformat(), product, round(sales, 2), round(production, 2), round(purchase, 2), round(stock, 2), round(previous, 2), round(future, 2)])
    return pd.DataFrame(rows, columns=["Date", "Product", "Sales/Usage", "Production", "Purchase Quantity", "Current Stock", "Previous Demand", "Future Demand"])


def prepare_data():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        df = create_synthetic_dataset()
        df.to_csv(DATA_PATH, index=False)
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    return df


def train():
    df = prepare_data()
    feature_columns = ["Product", "Sales/Usage", "Production", "Purchase Quantity", "Current Stock", "Previous Demand", "Month", "Quarter"]
    target = "Future Demand"
    X, y = df[feature_columns], df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)
    categorical = ["Product"]
    numeric = [c for c in feature_columns if c not in categorical]
    preprocessor = ColumnTransformer([("product", OneHotEncoder(handle_unknown="ignore"), categorical)], remainder="passthrough")
    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=180, max_depth=10, random_state=RANDOM_STATE),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=140, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE),
    }
    results = []
    predictions = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        predictions[name] = pred
        results.append({"model": name, "mae": round(mean_absolute_error(y_test, pred), 3), "rmse": round(np.sqrt(mean_squared_error(y_test, pred)), 3), "r2": round(r2_score(y_test, pred), 3)})
    results_df = pd.DataFrame(results).sort_values("rmse")
    best_name = results_df.iloc[0]["model"]
    best_pipeline = Pipeline([("preprocessor", preprocessor), ("model", candidates[best_name])])
    best_pipeline.fit(X, y)
    metadata = {
        "feature_columns": feature_columns,
        "target": target,
        "metrics": results_df.to_dict(orient="records"),
        "best_model": best_name,
        "test_actual": y_test.tolist(),
        "test_predicted": predictions[best_name].tolist(),
        "test_products": X_test["Product"].tolist(),
        "synthetic_data": True,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump({"pipeline": best_pipeline, "metadata": metadata}, f)
    print(f"Saved {DATA_PATH} ({len(df)} rows)")
    print(results_df.to_string(index=False))
    print(f"Best model: {best_name}")


if __name__ == "__main__":
    train()
