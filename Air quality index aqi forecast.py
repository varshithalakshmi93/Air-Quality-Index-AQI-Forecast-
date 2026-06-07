# ─────────────────────────────────────────────
# 4. IMPROVED MODEL TRAINING
# ─────────────────────────────────────────────
print("\n[4/7] Training Optimized Models …")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import ExtraTreesRegressor

results = {}

# Time Series Cross Validation
tscv = TimeSeriesSplit(n_splits=5)

# ── Extra Trees Regressor (usually performs very well)
print("   → Extra Trees Regressor …")
etr = ExtraTreesRegressor(
    n_estimators=500,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=False,
    random_state=42,
    n_jobs=-1
)

etr.fit(X_train_imp, y_train)
etr_pred = etr.predict(X_test_imp)

results["Extra Trees"] = {
    "model": etr,
    "pred": etr_pred,
    "MAE": mean_absolute_error(y_test, etr_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, etr_pred)),
    "R2": r2_score(y_test, etr_pred),
    "MAPE": mean_absolute_percentage_error(y_test, etr_pred) * 100
}

print(
    f"     MAE={results['Extra Trees']['MAE']:.2f} "
    f"RMSE={results['Extra Trees']['RMSE']:.2f} "
    f"R²={results['Extra Trees']['R2']:.4f}"
)

# ── Optimized Random Forest
print("   → Optimized Random Forest …")
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=25,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_imp, y_train)
rf_pred = rf.predict(X_test_imp)

results["Random Forest"] = {
    "model": rf,
    "pred": rf_pred,
    "MAE": mean_absolute_error(y_test, rf_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, rf_pred)),
    "R2": r2_score(y_test, rf_pred),
    "MAPE": mean_absolute_percentage_error(y_test, rf_pred) * 100
}

print(
    f"     MAE={results['Random Forest']['MAE']:.2f} "
    f"RMSE={results['Random Forest']['RMSE']:.2f} "
    f"R²={results['Random Forest']['R2']:.4f}"
)

# ── Optimized XGBoost
print("   → Optimized XGBoost …")

xgb = XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=2,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.5,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42
)

xgb.fit(
    X_train_imp,
    y_train,
    eval_set=[(X_test_imp, y_test)],
    verbose=False
)

xgb_pred = xgb.predict(X_test_imp)

results["XGBoost"] = {
    "model": xgb,
    "pred": xgb_pred,
    "MAE": mean_absolute_error(y_test, xgb_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, xgb_pred)),
    "R2": r2_score(y_test, xgb_pred),
    "MAPE": mean_absolute_percentage_error(y_test, xgb_pred) * 100
}

print(
    f"     MAE={results['XGBoost']['MAE']:.2f} "
    f"RMSE={results['XGBoost']['RMSE']:.2f} "
    f"R²={results['XGBoost']['R2']:.4f}"
)

# ── Ensemble Prediction (Best Accuracy)
print("   → Ensemble Model …")

ensemble_pred = (
    0.4 * rf_pred +
    0.4 * xgb_pred +
    0.2 * etr_pred
)

results["Ensemble"] = {
    "pred": ensemble_pred,
    "MAE": mean_absolute_error(y_test, ensemble_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, ensemble_pred)),
    "R2": r2_score(y_test, ensemble_pred),
    "MAPE": mean_absolute_percentage_error(y_test, ensemble_pred) * 100
}

print(
    f"     MAE={results['Ensemble']['MAE']:.2f} "
    f"RMSE={results['Ensemble']['RMSE']:.2f} "
    f"R²={results['Ensemble']['R2']:.4f}"
)
