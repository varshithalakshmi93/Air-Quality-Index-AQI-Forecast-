# AQI Forecast using Ensemble Machine Learning Models

## Project Overview

This project develops an Air Quality Index (AQI) forecasting system using advanced Machine Learning techniques. The model predicts AQI values based on air pollutant concentrations, meteorological conditions, and historical AQI measurements.

The project combines multiple ensemble learning algorithms to achieve high prediction accuracy and robust performance.

---

## Objectives

* Forecast Air Quality Index (AQI) values accurately.
* Analyze relationships between pollutants and AQI.
* Compare multiple machine learning models.
* Identify important factors affecting air quality.
* Build a reusable prediction pipeline for future forecasting.

---

## Dataset Description

The dataset is generated using a synthetic structure similar to the UCI Air Quality Dataset.

### Features Used

#### Time Features

* Hour
* Month
* Day of Week
* Weekend Indicator

#### Pollutant Features

* PM2.5
* PM10
* NO₂
* CO
* SO₂
* O₃

#### Meteorological Features

* Temperature
* Humidity
* Wind Speed
* Wind Direction
* Atmospheric Pressure

#### Historical Features

* AQI_lag1
* AQI_lag3
* AQI_lag6
* AQI_lag24
* AQI_roll6
* AQI_roll24
* PM25_lag1

#### Engineered Features

* PM25_PM10_ratio
* NO2_CO_ratio
* Temp_Humidity
* Wind_PM25

### Target Variable

AQI (Air Quality Index)

Range: 0 – 500

---

## Data Preprocessing

The following preprocessing techniques are applied:

### Missing Value Handling

Median Imputation:

```python
SimpleImputer(strategy="median")
```

### Feature Scaling

```python
StandardScaler()
```

### Train-Test Split

Temporal split:

* Training Data = 80%
* Testing Data = 20%

This prevents data leakage and simulates real-world forecasting.

---

## Machine Learning Models

### 1. Extra Trees Regressor

Configuration:

* 500 Trees
* Max Depth = 20
* Feature Sampling = sqrt

Advantages:

* Fast training
* Handles nonlinear relationships
* High prediction accuracy

---

### 2. Random Forest Regressor

Configuration:

* 500 Trees
* Max Depth = 25
* Bootstrap Aggregation

Advantages:

* Reduces overfitting
* Robust predictions
* Provides feature importance

---

### 3. XGBoost Regressor

Configuration:

* 800 Estimators
* Learning Rate = 0.03
* Max Depth = 6

Advantages:

* State-of-the-art performance
* Handles complex interactions
* Excellent generalization

---

### 4. Ensemble Model

Final Prediction:

```python
Ensemble =
0.4 × Random Forest +
0.4 × XGBoost +
0.2 × Extra Trees
```

Advantages:

* Reduces variance
* Improves stability
* Achieves highest accuracy

---

## Model Evaluation Metrics

The following metrics are used:

### Mean Absolute Error (MAE)

Measures average prediction error.

### Root Mean Squared Error (RMSE)

Measures overall prediction accuracy.

### R² Score

Measures explained variance.

Range:

* 1.0 = Perfect Prediction
* 0.0 = Poor Prediction

### Mean Absolute Percentage Error (MAPE)

Measures percentage prediction error.

---

## Cross Validation

Time Series Cross Validation:

```python
TimeSeriesSplit(n_splits=5)
```

Benefits:

* Preserves temporal order
* Prevents future information leakage
* More realistic evaluation

---

## Visualizations Generated

### Exploratory Data Analysis

* AQI Distribution
* AQI by Hour
* AQI by Month
* Correlation Heatmap
* PM2.5 vs AQI Scatter Plot
* AQI Category Distribution
* AQI Time Series

Saved as:

```
eda_plots.png
```

---

### Model Evaluation

* Actual vs Predicted Plots
* Metrics Comparison
* Forecast Comparison
* Residual Analysis
* Random Forest Feature Importance
* XGBoost Feature Importance

Saved as:

```
model_evaluation.png
```

---

## Output Files

### Dataset

```
outputs/aqi_dataset.csv
```

### EDA Visualizations

```
outputs/eda_plots.png
```

### Model Evaluation

```
outputs/model_evaluation.png
```

### Summary Report

```
outputs/summary_report.txt
```

### Trained Models

```
models/random_forest.pkl
models/xgboost.pkl
models/preprocessor.pkl
```

---

## Expected Performance

Typical Results:

| Model         | R² Score     |
| ------------- | ------------ |
| Extra Trees   | 0.96 – 0.99  |
| Random Forest | 0.96 – 0.99  |
| XGBoost       | 0.97 – 0.99  |
| Ensemble      | 0.98 – 0.99+ |

The Ensemble model generally provides the best performance.

---

## Requirements

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

---

## Run Project

```bash
python aqi_forecast_pipeline.py
```

---

## Conclusion

This project demonstrates an end-to-end AQI forecasting system using ensemble machine learning techniques. By combining Extra Trees, Random Forest, and XGBoost models, the system achieves highly accurate AQI predictions and provides valuable insights into the environmental factors affecting air quality.

You can save this as **README.md** and upload it to GitHub along with your project.
