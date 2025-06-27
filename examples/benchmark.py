# examples/benchmark.py
##### you need to pip install xgboost

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import os

# --- Import Models ---
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# --- Import from your library ---
from brinias import Brinias
from brinias.file_utils import fit_and_transform_data, transform_new_data

# === 1. Configuration ===

TRAIN_FILE = "dataeth.csv"
TEST_FILE = "dataeth2.csv"
TARGET_COLUMN = "next_close"
MODEL_OUTPUT_DIR = "brinias_benchmark_model" 
print("🚀 Starting Fair Benchmark...")
print(f"Training data: {TRAIN_FILE}")
print(f"Testing data:  {TEST_FILE}")

# === 2. Load and Prepare Data ===
print("\nProcessing training data...")
X_train, y_train, feature_names, task_type = fit_and_transform_data(
    TRAIN_FILE, TARGET_COLUMN, MODEL_OUTPUT_DIR
)

print("Processing test data...")
df_test_raw = pd.read_csv(TEST_FILE)
X_test = transform_new_data(df_test_raw, MODEL_OUTPUT_DIR)
y_test = df_test_raw[TARGET_COLUMN].values

print(f"\nTraining on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples.")
print(f"Number of features: {X_train.shape[1]}")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state=42, n_jobs=-1),
    "Brinias": Brinias(
        n_features=X_train.shape[1],
        feature_names=feature_names,
        task=task_type,
        generations=120,
        pop_size=100,
        cv_folds=15,
        seed=42
    )
}

# === 4. Run Benchmark ===
results = {}
predictions_dict = {} 
all_preds_df = pd.DataFrame({'y_true': y_test})

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"Training finished in {train_time:.2f} seconds.")
    
    y_pred = model.predict(X_test)
    all_preds_df[name] = y_pred
    results[name] = {"Time (s)": train_time}

# === 5. Clean Data and Calculate Metrics ===
print("\nCleaning results and calculating final metrics...")

original_rows = len(all_preds_df)
all_preds_df.replace([np.inf, -np.inf], np.nan, inplace=True)
cleaned_preds_df = all_preds_df.dropna()
dropped_rows = original_rows - len(cleaned_preds_df)

if dropped_rows > 0:
    print(f"⚠️ Dropped {dropped_rows} rows due to NaN values in target or predictions.")

y_test_clean = cleaned_preds_df['y_true']
for name in models.keys():
    y_pred_clean = cleaned_preds_df[name]
    results[name]["MSE"] = mean_squared_error(y_test_clean, y_pred_clean)
    results[name]["R2"] = r2_score(y_test_clean, y_pred_clean)
    predictions_dict[name] = y_pred_clean

# === 6. Report Results ===
print("\n\n--- BENCHMARK RESULTS ---")
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by="MSE", ascending=True)
print(results_df)

# === 7. Plotting ===
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(16, 9))

plt.plot(y_test_clean.values, label="Actual Values", color='black', linewidth=3, alpha=0.8)

colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
for i, (name, preds) in enumerate(predictions_dict.items()):
    plt.plot(preds.values, label=f"{name} Predictions", color=colors[i], linestyle='--', alpha=0.9)

plt.title(f"Benchmark: Actual vs. Predicted '{TARGET_COLUMN}' on Cleaned Test Data", fontsize=16)
plt.xlabel("Sample Index in Cleaned Test Set", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()