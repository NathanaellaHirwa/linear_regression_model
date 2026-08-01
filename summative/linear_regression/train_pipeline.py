"""
Crop Yield Prediction - Linear Regression Summative
Mission: Support food-security planning in Africa (and globally) by predicting
crop yield (hg/ha) from climate and agri-input variables, so agencies like
FAO-aligned ministries and NGOs can forecast production ahead of the season.

Dataset: FAO / World Bank crop yield data (Area, Item, Year, rainfall,
pesticides, avg temperature -> hg/ha_yield). 28,242 rows, 101 countries,
10 crops (includes Rwanda + Cassava, Maize, Potatoes, etc).
Source: https://github.com/ManikantaSanjay/crop_yield_prediction_regression
(originally aggregated from FAOSTAT and World Bank Climate data)
"""

import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
df = pd.read_csv("yield_df.csv")
if df.columns[0].startswith("Unnamed"):
    df = df.drop(columns=[df.columns[0]])

print("Raw shape:", df.shape)
print(df.isnull().sum())

# Basic cleaning: drop exact duplicate rows, drop any remaining nulls
df = df.drop_duplicates()
df = df.dropna()
print("After cleaning:", df.shape)

# ---------------------------------------------------------------------------
# 2. VISUALIZATIONS + INTERPRETATION
# ---------------------------------------------------------------------------
numeric_cols = ["Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp", "hg/ha_yield"]

# --- Correlation heatmap ---
plt.figure(figsize=(7, 5))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Correlation Heatmap - Numeric Features vs Yield")
plt.tight_layout()
plt.savefig("plots_correlation_heatmap.png", dpi=120)
plt.close()

# Interpretation (printed + saved to a text summary used in README/notebook markdown):
# pesticides_tonnes has the strongest positive correlation with yield, meaning
# higher pesticide use is associated with higher recorded yield in this
# dataset (likely a proxy for larger, more intensive farming operations).
# average_rain_fall_mm_per_year has a weak/inconsistent correlation - yield
# depends heavily on WHICH crop/region it falls in, not rainfall alone.
# avg_temp has a mild negative relationship with yield.

# --- Distribution plots ---
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
sns.histplot(df["hg/ha_yield"], bins=50, ax=axes[0, 0], color="#3D4F7C")
axes[0, 0].set_title("Distribution of Crop Yield (hg/ha)")
sns.histplot(df["average_rain_fall_mm_per_year"], bins=50, ax=axes[0, 1], color="#4C956C")
axes[0, 1].set_title("Distribution of Average Rainfall (mm/yr)")
sns.histplot(df["pesticides_tonnes"], bins=50, ax=axes[1, 0], color="#D68C45")
axes[1, 0].set_title("Distribution of Pesticide Use (tonnes)")
sns.histplot(df["avg_temp"], bins=50, ax=axes[1, 1], color="#A63A50")
axes[1, 1].set_title("Distribution of Average Temperature (C)")
plt.tight_layout()
plt.savefig("plots_distributions.png", dpi=120)
plt.close()

# --- Top crops by average yield (bar) ---
plt.figure(figsize=(9, 5))
top_items = df.groupby("Item")["hg/ha_yield"].mean().sort_values(ascending=False)
sns.barplot(x=top_items.values, y=top_items.index, palette="viridis")
plt.xlabel("Average Yield (hg/ha)")
plt.title("Average Yield by Crop Type")
plt.tight_layout()
plt.savefig("plots_yield_by_crop.png", dpi=120)
plt.close()

print("\nInterpretation notes:")
print("- Crop type (Item) is the single biggest driver of yield scale (potatoes/cassava")
print("  yield in hg/ha dwarfs cereals like wheat/maize) -> Item MUST be encoded and kept.")
print("- pesticides_tonnes correlates positively with yield -> keep, meaningful weight.")
print("- Year has a mild upward trend (tech/practice improvements over time) -> keep.")
print("- Area (country) captures soil/policy/infrastructure differences -> keep but")
print("  one-hot encoded; high-cardinality, so tree models will value it more than linear.")

# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
FEATURES = ["Area", "Item", "Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
TARGET = "hg/ha_yield"

X = df[FEATURES].copy()
y = df[TARGET].copy()

categorical_features = ["Area", "Item"]
numeric_features = ["Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]

# Numeric -> standardized ; Categorical -> one-hot encoded (handles the
# "convert to numeric" + "standardize" requirements in one preprocessing step)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print("\nTrain/Test shapes:", X_train.shape, X_test.shape)

# ---------------------------------------------------------------------------
# 4. MODEL TRAINING - 4 IMPLEMENTATIONS
#    (a) SGDRegressor       -> the "stochastic" gradient-descent linear model
#    (b) LinearRegression   -> classic OLS linear model (comparison baseline)
#    (c) DecisionTreeRegressor
#    (d) RandomForestRegressor
# ---------------------------------------------------------------------------
models = {
    "SGD_LinearRegression": SGDRegressor(
        max_iter=1, warm_start=True, learning_rate="invscaling",
        eta0=0.01, random_state=RANDOM_STATE
    ),
    "OLS_LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
    "RandomForest": RandomForestRegressor(
        n_estimators=80, max_depth=12, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
    ),
}

results = {}
pipelines = {}

# --- (a) SGD trained epoch-by-epoch so we can plot a real loss curve ---
sgd_pipeline_pre = Pipeline([("prep", preprocessor)])
X_train_t = sgd_pipeline_pre.fit_transform(X_train)
X_test_t = sgd_pipeline_pre.transform(X_test)

sgd = models["SGD_LinearRegression"]
n_epochs = 60
train_losses, test_losses = [], []
for epoch in range(n_epochs):
    sgd.partial_fit(X_train_t, y_train)
    train_pred = sgd.predict(X_train_t)
    test_pred = sgd.predict(X_test_t)
    train_losses.append(mean_squared_error(y_train, train_pred))
    test_losses.append(mean_squared_error(y_test, test_pred))

plt.figure(figsize=(8, 5))
plt.plot(range(1, n_epochs + 1), train_losses, label="Train Loss (MSE)")
plt.plot(range(1, n_epochs + 1), test_losses, label="Test Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("SGD Linear Regression - Loss Curve (Train vs Test)")
plt.legend()
plt.tight_layout()
plt.savefig("plots_loss_curve.png", dpi=120)
plt.close()

pipelines["SGD_LinearRegression"] = Pipeline([("prep", preprocessor), ("model", sgd)])
# refit prep+model cleanly for consistent pipeline object (sgd already fitted via partial_fit)
pipelines["SGD_LinearRegression"].named_steps["prep"] = sgd_pipeline_pre.named_steps["prep"]

sgd_train_pred = sgd.predict(X_train_t)
sgd_test_pred = sgd.predict(X_test_t)
results["SGD_LinearRegression"] = {
    "train_mse": mean_squared_error(y_train, sgd_train_pred),
    "test_mse": mean_squared_error(y_test, sgd_test_pred),
    "test_r2": r2_score(y_test, sgd_test_pred),
}

# --- (b), (c), (d): fit normally via full pipelines ---
for name in ["OLS_LinearRegression", "DecisionTree", "RandomForest"]:
    pipe = Pipeline([("prep", preprocessor), ("model", models[name])])
    pipe.fit(X_train, y_train)
    train_pred = pipe.predict(X_train)
    test_pred = pipe.predict(X_test)
    results[name] = {
        "train_mse": mean_squared_error(y_train, train_pred),
        "test_mse": mean_squared_error(y_test, test_pred),
        "test_r2": r2_score(y_test, test_pred),
    }
    pipelines[name] = pipe

print("\n=== MODEL COMPARISON (lower MSE / higher R2 is better) ===")
results_df = pd.DataFrame(results).T.sort_values("test_mse")
print(results_df)
results_df.to_csv("model_comparison_results.csv")

best_model_name = results_df["test_mse"].idxmin()
print(f"\nBest performing model by Test MSE: {best_model_name}")

# ---------------------------------------------------------------------------
# 5. SCATTER PLOT - BEST FIT LINE (Rainfall vs Yield, before/after OLS fit)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sample = df.sample(min(3000, len(df)), random_state=RANDOM_STATE)

axes[0].scatter(sample["average_rain_fall_mm_per_year"], sample["hg/ha_yield"],
                 alpha=0.3, s=10, color="gray")
axes[0].set_title("Before Training: Raw Rainfall vs Yield")
axes[0].set_xlabel("Average Rainfall (mm/yr)")
axes[0].set_ylabel("Yield (hg/ha)")

# simple 1-feature OLS just for this illustrative best-fit line
simple_lr = LinearRegression()
x_simple = sample[["average_rain_fall_mm_per_year"]].values
y_simple = sample["hg/ha_yield"].values
simple_lr.fit(x_simple, y_simple)
x_line = np.linspace(x_simple.min(), x_simple.max(), 100).reshape(-1, 1)
y_line = simple_lr.predict(x_line)

axes[1].scatter(sample["average_rain_fall_mm_per_year"], sample["hg/ha_yield"],
                 alpha=0.3, s=10, color="gray")
axes[1].plot(x_line, y_line, color="#A63A50", linewidth=3)
axes[1].set_title("After Training: Best-Fit Line (OLS)")
axes[1].set_xlabel("Average Rainfall (mm/yr)")
axes[1].set_ylabel("Yield (hg/ha)")
plt.tight_layout()
plt.savefig("plots_scatter_bestfit.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# 6. SAVE BEST MODEL
# ---------------------------------------------------------------------------
best_pipeline = pipelines[best_model_name]
joblib.dump(best_pipeline, "best_model.joblib")

metadata = {
    "best_model_name": best_model_name,
    "features": FEATURES,
    "target": TARGET,
    "categorical_options": {
        "Area": sorted(df["Area"].unique().tolist()),
        "Item": sorted(df["Item"].unique().tolist()),
    },
    "numeric_ranges": {
        "Year": [int(df["Year"].min()), int(df["Year"].max())],
        "average_rain_fall_mm_per_year": [float(df["average_rain_fall_mm_per_year"].min()), float(df["average_rain_fall_mm_per_year"].max())],
        "pesticides_tonnes": [float(df["pesticides_tonnes"].min()), float(df["pesticides_tonnes"].max())],
        "avg_temp": [float(df["avg_temp"].min()), float(df["avg_temp"].max())],
    },
    "test_mse": float(results_df.loc[best_model_name, "test_mse"]),
    "test_r2": float(results_df.loc[best_model_name, "test_r2"]),
}
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved: best_model.joblib, model_metadata.json, model_comparison_results.csv")

# ---------------------------------------------------------------------------
# 7. PREDICTION SCRIPT (single row from test set) -> feeds Task 2 API
# ---------------------------------------------------------------------------
sample_row = X_test.iloc[[0]]
actual_value = y_test.iloc[0]
predicted_value = best_pipeline.predict(sample_row)[0]

print("\n=== SAMPLE PREDICTION (Task 2 hand-off) ===")
print("Input row:")
print(sample_row.to_dict(orient="records")[0])
print(f"Actual yield:    {actual_value:.2f} hg/ha")
print(f"Predicted yield: {predicted_value:.2f} hg/ha")
