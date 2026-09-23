from pathlib import Path
import base64
import io
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cotton_inventory.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
app = Flask(__name__)


def load_assets():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    with MODEL_PATH.open("rb") as f:
        bundle = pickle.load(f)
    return df, bundle


def chart_uri(fig):
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_charts(df, metadata):
    charts = {}
    trends = df.groupby("Date", as_index=False)["Future Demand"].mean()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(trends["Date"], trends["Future Demand"], color="#176b87", linewidth=2.5)
    ax.set_title("Historical average demand trend")
    ax.set_ylabel("Demand quantity")
    ax.grid(alpha=.2)
    charts["trend"] = chart_uri(fig)

    product = df.groupby("Product")["Future Demand"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    product.plot(kind="barh", ax=ax, color="#55a630")
    ax.set_title("Average demand by product")
    ax.set_xlabel("Demand quantity")
    ax.grid(axis="x", alpha=.2)
    charts["product"] = chart_uri(fig)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.scatter(metadata["test_actual"], metadata["test_predicted"], alpha=.7, color="#e76f51")
    lo, hi = min(metadata["test_actual"]), max(metadata["test_actual"])
    ax.plot([lo, hi], [lo, hi], "--", color="#264653")
    ax.set_title("Actual vs predicted demand (test set)")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(alpha=.2)
    charts["actual_predicted"] = chart_uri(fig)

    sample = df.groupby("Product")[["Current Stock", "Future Demand"]].mean()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    sample.plot(kind="bar", ax=ax, color=["#2a9d8f", "#e9c46a"])
    ax.set_title("Average current stock vs future demand")
    ax.set_ylabel("Quantity")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=.2)
    charts["stock_demand"] = chart_uri(fig)

    status_counts = pd.Series({"Sufficient": 0, "Low": 0, "Critical": 0})
    statuses = df.apply(lambda r: inventory_status(r["Future Demand"], r["Current Stock"]), axis=1)
    status_counts = statuses.value_counts().reindex(status_counts.index, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(status_counts.index, status_counts.values, color=["#2a9d8f", "#f4a261", "#e76f51"])
    ax.set_title("Inventory-status distribution")
    ax.set_ylabel("Number of records")
    charts["status"] = chart_uri(fig)
    return charts


def inventory_status(predicted, stock):
    ratio = stock / predicted if predicted else 0
    if ratio < 0.60:
        return "Critical"
    if ratio < 1.00:
        return "Low"
    return "Sufficient"


def prediction_from_form(form, bundle):
    date = pd.to_datetime(form.get("date") or pd.Timestamp.today().date())
    row = pd.DataFrame([{
        "Product": form.get("product", "Raw Cotton"),
        "Sales/Usage": float(form.get("sales_usage", 300)),
        "Production": float(form.get("production", 300)),
        "Purchase Quantity": float(form.get("purchase_quantity", 80)),
        "Current Stock": float(form.get("current_stock", 300)),
        "Previous Demand": float(form.get("previous_demand", 300)),
        "Month": date.month,
        "Quarter": date.quarter,
    }])
    predicted = max(0, float(bundle["pipeline"].predict(row)[0]))
    stock = row.loc[0, "Current Stock"]
    status = inventory_status(predicted, stock)
    shortage = max(0, predicted - stock)
    return round(predicted, 2), status, round(shortage, 2)


@app.route("/", methods=["GET", "POST"])
def index():
    df, bundle = load_assets()
    prediction = None
    if request.method == "POST":
        try:
            prediction = prediction_from_form(request.form, bundle)
        except (TypeError, ValueError):
            prediction = (None, "Please enter valid numeric values.", None)
    charts = build_charts(df, bundle["metadata"])
    metrics = bundle["metadata"]["metrics"]
    overview = {
        "rows": len(df), "products": df["Product"].nunique(),
        "best_model": bundle["metadata"]["best_model"],
        "avg_stock": round(df["Current Stock"].mean(), 1),
    }
    return render_template("index.html", products=sorted(df["Product"].unique()), charts=charts, metrics=metrics, overview=overview, prediction=prediction, form=request.form)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
