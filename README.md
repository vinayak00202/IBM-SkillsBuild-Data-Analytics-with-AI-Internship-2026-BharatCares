# House Price Prediction — Internship Project

**Author:** Vinayak Ojha 

## Overview
This project builds and evaluates machine learning models that predict house
prices from a set of structural and locational features (area, number of
bedrooms/bathrooms, stories, parking, and amenities such as air conditioning,
a main-road location, and a preferred area).

## Project Structure
```
HousePricePrediction_NitinKumarSaini/
├── HousePricePrediction_Analysis.ipynb   # Main analysis notebook (executed, with outputs)
├── Housing.csv                           # Raw dataset (200 rows, 13 columns)
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
└── charts/
    ├── chart1_histogram.png              # Distribution of house prices
    ├── chart2_heatmap.png                # Feature correlation with price
    └── chart3_scatter.png                # Actual vs. predicted prices (test set)
```

## Dataset
- **Source file:** `Housing.csv`
- **Rows / columns:** 200 rows × 13 columns
- **Target variable:** `price`
- **Features:** `area`, `bedrooms`, `bathrooms`, `stories`, `mainroad`,
  `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `parking`,
  `prefarea`, `furnishingstatus`
- **Missing values:** none found after cleaning (`dropna()` / `drop_duplicates()`
  were applied as a precaution)

## Pipeline (Notebook Tasks)
1. **Data Loading & Exploration** — load the CSV, inspect shape, check for
   missing values.
2. **Data Cleaning** — drop nulls/duplicates, one-hot encode the categorical
   columns (`mainroad`, `guestroom`, `basement`, `hotwaterheating`,
   `airconditioning`, `prefarea`, `furnishingstatus`).
3. **Model Building** — an 80/20 train/test split, then fit and evaluate:
   - `LinearRegression`
   - `RandomForestRegressor`
4. **Visualization** — a price-distribution histogram, a feature-correlation
   heatmap, and an actual-vs-predicted scatter plot.
5. **Insights & Summary** — written interpretation of what drives price in
   this dataset.

## Results
Metrics from the notebook run (80/20 train/test split, `random_state=42`):

| Model              | MAE        | RMSE       | R² |
|---------------------|-----------:|-----------:|-----:|
| Linear Regression    | 108,793     | 137,290     | 0.852 |
| Random Forest         | 132,580     | 181,662     | 0.741 |

On this dataset, **Linear Regression generalized better on the held-out test
set** than Random Forest (higher R², lower error). This is a plausible
outcome on a small dataset (200 rows): the relationship between area/rooms
and price is fairly linear, and a Random Forest's extra flexibility can
overfit the training split without added benefit here. With more data or
hyperparameter tuning (e.g. limiting tree depth, tuning `n_estimators`), the
Random Forest could close or reverse this gap.

The strongest correlates of price (see `chart2_heatmap.png`) are **area**,
**number of bathrooms**, **air conditioning**, and **parking availability**;
`mainroad` location also has a noticeable positive association.

## Recommendation for Real Estate Business
Marketing and pricing strategy should emphasize properties with air
conditioning, generous area, and a main-road / preferred-area location, since
these features are the strongest empirical drivers of higher valuations in
this dataset. Agents can use the trained regression model as a quick
sanity-check to flag listings that appear under- or over-valued relative to
their features.

## How to Run
1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter and run the notebook top to bottom:
   ```bash
   jupyter notebook HousePricePrediction_Analysis.ipynb
   ```
   The notebook will re-generate the three charts into the `charts/` folder
   and reprint the evaluation metrics above.

## Notes / Next Steps
- Try regularized linear models (Ridge/Lasso) and gradient-boosted trees
  (e.g. `GradientBoostingRegressor`, XGBoost) to see if they beat plain
  Linear Regression.
- Use `GridSearchCV`/`RandomizedSearchCV` to tune the Random Forest
  (`n_estimators`, `max_depth`, `min_samples_leaf`).
- With only 200 rows, k-fold cross-validation would give a more reliable
  performance estimate than a single train/test split.
