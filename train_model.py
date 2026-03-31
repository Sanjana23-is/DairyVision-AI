"""
train_model.py
Trains multiple regression models (Linear, Decision Tree, Random Forest, XGBoost) 
on the synthetic dairy data. Evaluates performance, compares them, and saves 
the primary models and comparison results.
"""
import pandas as pd
import numpy as np
import time
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed. It will be skipped in comparison.")

def load_and_preprocess_data(filepath='dairy_data.csv'):
    """ Loads data and creates the preprocessing pipeline. """
    df = pd.read_csv(filepath)
    
    # Feature matrix X and target y
    X = df.drop(columns=['cow_id', 'milk_yield'])
    y = df['milk_yield']
    
    # Identify numerical and categorical features
    numerical_features = ['age', 'feed_quantity', 'temperature', 'humidity', 'lactation_stage']
    categorical_features = ['breed']
    
    # Preprocessing pipeline ensures fair comparison amongst all models
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # 80-20 Train-test split
    return train_test_split(X, y, test_size=0.2, random_state=42), preprocessor

def train_and_evaluate(name, pipeline, X_train, y_train, X_test, y_test):
    """
    Trains the given pipeline, predicts on the test set, evaluates metrics,
    and returns a dictionary of the results.
    """
    # Train the attached model within the pipeline
    pipeline.fit(X_train, y_train)
    
    # Measure inference time on test set
    start_time = time.time()
    y_pred = pipeline.predict(X_test)
    end_time = time.time()
    inference_time = end_time - start_time
    
    # Compute the requested metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print structured results to terminal
    print(f"\n--- Model: {name} ---")
    print(f"R² Score:       {r2:.4f}")
    print(f"RMSE:           {rmse:.4f} Liters")
    print(f"MAE:            {mae:.4f} Liters")
    print(f"Inference Time: {inference_time:.4f} seconds")
    
    # Return metrics for the final Pandas DataFrame comparison table
    return {
        'Model': name,
        'R² Score': r2,
        'RMSE': rmse,
        'MAE': mae,
        'Inference Time': inference_time,
        'Pipeline': pipeline  # return pipeline to save it later if needed
    }

def run_test_cases(model):
    """ Runs 3 specified sample test cases using the primary model. """
    print("\n--- Running 3 Sample Test Cases (Using Primary Model) ---")
    test_cases = pd.DataFrame([
        # Test Case 1: Ideal conditions
        {'age': 5, 'breed': 'Holstein', 'feed_quantity': 25.0, 'temperature': 15.0, 'humidity': 50.0, 'lactation_stage': 60},
        # Test Case 2: Extreme Heat Stress
        {'age': 5, 'breed': 'Holstein', 'feed_quantity': 25.0, 'temperature': 38.0, 'humidity': 85.0, 'lactation_stage': 60},
        # Test Case 3: Late Lactation, Low Feed
        {'age': 5, 'breed': 'Holstein', 'feed_quantity': 15.0, 'temperature': 15.0, 'humidity': 50.0, 'lactation_stage': 300}
    ])
    
    predictions = model.predict(test_cases)
    
    print("\nTest Case 1 (Ideal Conditions): Expected high yield.")
    print(f"Prediction: {predictions[0]:.2f} Liters")
    
    print("\nTest Case 2 (Heat Stress): Expected significant yield drop.")
    print(f"Prediction: {predictions[1]:.2f} Liters")
    
    print("\nTest Case 3 (Late Lactation, Low Feed): Expected lowest yield.")
    print(f"Prediction: {predictions[2]:.2f} Liters")

def main():
    print("Loading data and setting up preprocessing...")
    (X_train, X_test, y_train, y_test), preprocessor = load_and_preprocess_data()
    
    # Dictionary to hold the regressor model definitions
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    }
    
    # Dynamically Add XGBoost if available
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)

    results_list = []
    primary_model = None
    
    # Iteratively train, predict, and evaluate all models using the exact same preprocessing
    for name, regressor in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ])
        
        # Train and collect metrics
        metrics = train_and_evaluate(name, pipeline, X_train, y_train, X_test, y_test)
        results_list.append(metrics)
        
        # Save specific models to maintain compatibility with Streamlit app and system design
        if name == "Random Forest":
            rf_model_path = 'random_forest_model.joblib'
            joblib.dump(pipeline, rf_model_path)
            print(f">>> Random Forest model saved to {rf_model_path}")
            primary_model = pipeline
            
        elif name == "XGBoost":
            xgb_model_path = 'xgboost_model.joblib'
            joblib.dump(pipeline, xgb_model_path)
            print(f">>> XGBoost model saved to {xgb_model_path}")

    # Create final comparison dataframe
    print("\n=======================================================")
    print("FINAL MODEL COMPARISON")
    print("=======================================================")
    
    # Exclude the Pipeline object from the dataframe before display
    df_records = [{k: v for k, v in result.items() if k != 'Pipeline'} for result in results_list]
    comparison_df = pd.DataFrame(df_records)
    
    # Sort by R² Score in descending order
    comparison_df = comparison_df.sort_values(by='R² Score', ascending=False)
    
    # Print clearly formatted table to console
    print(comparison_df.to_string(index=False))
    
    # Save the comparison results to a CSV file
    csv_path = 'model_comparison_results.csv'
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nModel comparison results successfully saved to {csv_path}")
    
    # Run test cases using the primary model (Random Forest)
    if primary_model:
        run_test_cases(primary_model)

if __name__ == "__main__":
    main()
