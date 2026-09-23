# Machine Learning-Based Cotton Mill Inventory Management

A **Machine Learning + Flask** project that predicts next-period cotton-product demand and provides inventory status and reorder recommendations. The project is designed as a university educational prototype using synthetic data.

## Problem

Cotton mills need to avoid:

* **Stock shortages**
* **Overstocking**

The system predicts future demand and compares it with current stock to support inventory planning.

## Products

* Raw Cotton
* Cotton Yarn
* Cotton Fabric
* Cotton Thread
* Cotton Waste

## Machine Learning Workflow

```text
Dataset
→ Preprocessing
→ Feature Engineering
→ One-Hot Encoding
→ Train/Test Split
→ Model Training
→ Model Comparison
→ Best Model
→ Demand Prediction
→ Inventory Status
→ Reorder Recommendation
```

## Algorithms & Results

| Model                       |        MAE |       RMSE |        R² |
| --------------------------- | ---------: | ---------: | --------: |
| **Linear Regression**       | **11.981** | **15.096** | **0.973** |
| Random Forest Regressor     |     14.418 |     18.117 |     0.961 |
| Gradient Boosting Regressor |     15.225 |     19.071 |     0.956 |

**Selection:** The model with the lowest test RMSE is selected. For the included synthetic dataset, **Linear Regression** has the lowest RMSE.

## Inventory Logic

| Status         | Condition                            |
| -------------- | ------------------------------------ |
| **Sufficient** | Stock ≥ Predicted Demand             |
| **Low**        | Stock is 60%–99% of Predicted Demand |
| **Critical**   | Stock < 60% of Predicted Demand      |

**Reorder Quantity:**

```text
max(0, Predicted Demand - Current Stock)
```

## Dataset

Main features:

`Date` • `Product` • `Sales/Usage` • `Production` • `Purchase Quantity` • `Current Stock` • `Previous Demand`

**Target:** `Future Demand`

The project also creates **Month** and **Quarter** features and uses **One-Hot Encoding** for Product.

## Technologies

**Python • Pandas • NumPy • Scikit-learn • Flask • HTML • CSS • Matplotlib**

## Main Features

* Demand prediction
* Inventory status detection
* Reorder quantity calculation
* Model comparison
* Flask web dashboard
* Data visualization

## Project Structure

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
└── requirements.txt
```

## Run the Project

```bash
pip install -r requirements.txt
python train_model.py
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

