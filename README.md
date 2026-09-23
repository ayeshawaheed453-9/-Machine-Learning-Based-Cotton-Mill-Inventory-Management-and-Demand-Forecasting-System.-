# Machine Learning-Based Cotton Mill Inventory Management and Demand Forecasting System

A small, complete, beginner-friendly university ML project built with **Python, pandas, scikit-learn, Flask, HTML, CSS, and Matplotlib**. The system predicts next-period cotton-mill product demand and converts the prediction into an inventory status and reorder recommendation.

## Problem and purpose

Cotton mills need enough raw materials and finished goods to meet demand without tying up money in unnecessary stock. This project learns the relationship between recent mill activity and future demand, then compares predicted demand with current stock.

The five products are **Raw Cotton, Cotton Yarn, Cotton Fabric, Cotton Thread, and Cotton Waste**. The included CSV is synthetic and clearly labeled for educational use. It can be replaced with a real dataset using the same column names.

## Dataset columns

| Column | Meaning |
|---|---|
| Date | Observation date |
| Product | Cotton product name |
| Sales/Usage | Quantity sold or consumed |
| Production | Quantity produced |
| Purchase Quantity | Quantity purchased |
| Current Stock | Quantity currently available |
| Previous Demand | Demand in the previous period |
| Future Demand | Numeric regression target for the next period |

## Machine-learning workflow

`Dataset → Preprocessing → Feature Engineering → Train/Test Split → Model Training → Model Comparison → Best Model → Demand Prediction → Inventory Status → Reorder Recommendation`

`train_model.py` adds month and quarter features, one-hot encodes Product, and compares:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The models are evaluated with **MAE, RMSE, and R²**. The model with the lowest test RMSE is selected, retrained on all rows, and saved to `models/best_model.pkl`.

## Inventory logic

The app uses predicted demand and current stock:

- **Sufficient:** current stock is at least predicted demand.
- **Low:** current stock is between 60% and 99% of predicted demand.
- **Critical:** current stock is below 60% of predicted demand.
- **Reorder quantity:** `max(0, predicted demand - current stock)`.

These thresholds are simple teaching rules and should be adjusted with a mill's safety-stock policy in a production system.

## Run locally

```bash
cd cotton_inventory_ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Open <http://127.0.0.1:5000> in a browser. The training script is safe to rerun; it creates the dataset only when the CSV is absent and refreshes the model artifact.

## Project files

```text
cotton_inventory_ml/
├── app.py
├── train_model.py
├── data/
│   └── cotton_inventory.csv
├── models/
│   └── best_model.pkl
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── requirements.txt
└── README.md
```

## Presentation outline

1. Explain the stock-shortage and overstocking problem.
2. Show the dataset columns and synthetic-data note.
3. Explain feature engineering and one-hot encoding.
4. Compare the three regression models using MAE, RMSE, and R².
5. Demonstrate one form prediction in the Flask dashboard.
6. Interpret the inventory status and reorder quantity.
7. Discuss limitations: synthetic data, one-step forecast, and simple business thresholds.

## Limitations and responsible use

This is a compact educational prototype, not a production planning system. Real deployment should validate data quality, include lead time and safety stock, handle holidays and supply disruptions, monitor drift, and obtain approval from inventory managers before placing purchase orders.
