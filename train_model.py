"""
train_model.py
Trains Random Forest and XGBoost Regressors on the synthetic dairy data,
evaluates performance, and saves the models.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import time

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed. It will be skipped.")

def load_and_preprocess_data(filepath='dairy_data.csv'):
    df = pd.read_csv(filepath)
    
    # Feature matrix X and target y
    X = df.drop(columns=['cow_id', 'milk_yield'])
    y = df['milk_yield']
    
    # Identify numerical and categorical features
    numerical_features = ['age', 'feed_quantity', 'temperature', 'humidity', 'lactation_stage']
    categorical_features = ['breed']
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    return train_test_split(X, y, test_size=0.2, random_state=42), preprocessor

def evaluate_model(name, model, X_test, y_test, start_time):
    # Model predict inference time
    y_pred = model.predict(X_test)
    end_time = time.time()
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\\n--- {name} Evaluation ---")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE:     {rmse:.4f} Liters")
    print(f"MAE:      {mae:.4f} Liters")
    print(f"Inference Time (test set): {end_time - start_time:.4f} seconds")

def run_test_cases(model):
    print("\\n--- Running 3 Sample Test Cases ---")
    test_cases = pd.DataFrame([
        # Test Case 1: Ideal conditions
        {'age': 5, 'breed': 'Holstein', 'feed_quantity': 25.0, 'temperature': 15.0, 'humidity': 50.0, 'lactation_stage': 60},
        # Test Case 2: Extreme Heat Stress
        {'age': 5, 'breed': 'Holstein', 'feed_quantity': 25.0, 'temperature': 38.0, 'humidity': 85.0, 'lactation_stage': 60},
        # Test Case 3: Late Lactation, Low Feed
        {'age': 5, 'breed': 'Holstein', 'feed_quantity': 15.0, 'temperature': 15.0, 'humidity': 50.0, 'lactation_stage': 300}
    ])
    
    predictions = model.predict(test_cases)
    
    print("\\nTest Case 1 (Ideal Conditions): Expected high yield.")
    print(pd.DataFrame([test_cases.iloc[0]]))
    print(f"Prediction: {predictions[0]:.2f} Liters")
    
    print("\\nTest Case 2 (Heat Stress): Expected significant yield drop due to temp=38C, humidity=85%.")
    print(pd.DataFrame([test_cases.iloc[1]]))
    print(f"Prediction: {predictions[1]:.2f} Liters")
    
    print("\\nTest Case 3 (Late Lactation, Low Feed): Expected lowest yield due to high DIM and low feed.")
    print(pd.DataFrame([test_cases.iloc[2]]))
    print(f"Prediction: {predictions[2]:.2f} Liters")

def main():
    print("Loading data...")
    (X_train, X_test, y_train, y_test), preprocessor = load_and_preprocess_data()
    
    # 1. Random Forest (Primary Model)
    print("\\nTraining Primary Model: Random Forest Regressor...")
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
    ])
    
    start_time = time.time()
    rf_pipeline.fit(X_train, y_train)
    evaluate_model("Random Forest", rf_pipeline, X_test, y_test, start_time)
    
    rf_model_path = 'random_forest_model.joblib'
    joblib.dump(rf_pipeline, rf_model_path)
    print(f"Random Forest model saved to {rf_model_path}")

    # 2. XGBoost (Advanced / Future Enhancement)
    if XGB_AVAILABLE:
        print("\\nTraining Advanced Model: XGBoost Regressor...")
        xgb_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1))
        ])
        
        start_time = time.time()
        xgb_pipeline.fit(X_train, y_train)
        evaluate_model("XGBoost", xgb_pipeline, X_test, y_test, start_time)
        
        xgb_model_path = 'xgboost_model.joblib'
        joblib.dump(xgb_pipeline, xgb_model_path)
        print(f"XGBoost model saved to {xgb_model_path}")
        
    print("\\n--- End of Training ---")
    
    # Run test cases using the primary model
    run_test_cases(rf_pipeline)

if __name__ == "__main__":
    main()
