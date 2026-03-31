"""
app.py
Streamlit Dashboard for DairyVision AI.
Provides What-If Simulation and interactive charts.
"""
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# --- Config & Setup ---
st.set_page_config(page_title="DairyVision AI", layout="wide")
st.title("🐄 DairyVision AI: Smart Simulation & Prediction")

@st.cache_resource
def load_model():
    """ Load the Random Forest primary model """
    model_path = 'random_forest_model.joblib'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"Model {model_path} not found. Please run train_model.py first.")
        return None

@st.cache_data
def load_data():
    """ Load the generated dataset """
    data_path = 'dairy_data.csv'
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        st.error(f"Data {data_path} not found. Please run data_generation.py first.")
        return None

def predict_yield(model, input_data: dict) -> float:
    """
    What-If Simulation Function.
    Takes input parameters, structures them into a DataFrame, and predicts yield.
    Time Complexity: O(1) for a single inference with trained Random Forest (~1-5 milliseconds).
    """
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    return prediction

# --- Main App Logic ---
model = load_model()
df = load_data()

if model and df is not None:
    # Sidebar: What-If Simulation Controls
    st.sidebar.header("🎛️ What-If Simulation Engine")
    
    # Cow Selection
    unique_cows = df['cow_id'].unique()
    # Limit choices to a sample for UI responsiveness
    selected_cow = st.sidebar.selectbox("Select Cow Profile", unique_cows[:50])
    
    # Get current cow context
    cow_df = df[df['cow_id'] == selected_cow].iloc[0]
    
    st.sidebar.subheader("Adjust Parameters")
    
    # Sliders using the current cows physical parameters as default
    sim_feed = st.sidebar.slider("Feed Quantity (kg/day)", 10.0, 35.0, float(cow_df['feed_quantity']))
    sim_temp = st.sidebar.slider("Temperature (°C)", -10.0, 45.0, float(cow_df['temperature']))
    sim_humidity = st.sidebar.slider("Humidity (%)", 20.0, 100.0, float(cow_df['humidity']))
    sim_dim = st.sidebar.slider("Lactation Stage (Days In Milk)", 1, 400, int(cow_df['lactation_stage']))
    
    # --- Prediction Logic ---
    # Prepare input for simulation
    sim_input = {
        'age': cow_df['age'],
        'breed': cow_df['breed'],
        'feed_quantity': sim_feed,
        'temperature': sim_temp,
        'humidity': sim_humidity,
        'lactation_stage': sim_dim
    }
    
    predicted_yield = predict_yield(model, sim_input)
    
    # Calculate Base Prediction (using original unchanged dataset inputs)
    base_input = {
        'age': cow_df['age'],
        'breed': cow_df['breed'],
        'feed_quantity': cow_df['feed_quantity'],
        'temperature': cow_df['temperature'],
        'humidity': cow_df['humidity'],
        'lactation_stage': cow_df['lactation_stage']
    }
    base_predicted = predict_yield(model, base_input)
    delta = predicted_yield - base_predicted
    
    # --- UI Layout ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cow ID", cow_df['cow_id'])
    with col2:
        st.metric("Breed", cow_df['breed'])
    with col3:
        st.metric("Base Prediction", f"{base_predicted:.2f} L")
    with col4:
        st.metric("Simulated Predicted Yield", f"{predicted_yield:.2f} L", delta=f"{delta:.2f} L")

    st.markdown("---")
    
    # --- Visualizations ---
    st.subheader("📊 Farm Analytics & Trends")
    
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        # Plotly: Lactation Stage vs Milk Yield
        st.markdown("**Herd Yield by Lactation Stage**")
        # Sub-sample data to prevent browser lag on big plots
        sample_df = df.sample(min(2000, len(df)))
        fig_scatter = px.scatter(sample_df, x='lactation_stage', y='milk_yield', color='breed', 
                                 opacity=0.6, title="Yield Drop-off Over Lactation")
        # Add marker for the simulated cow
        fig_scatter.add_trace(go.Scatter(x=[sim_dim], y=[predicted_yield], mode='markers', 
                                         marker=dict(color='yellow', size=15, line=dict(color='black', width=2)),
                                         name='Simulated Cow'))
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col_v2:
        # Plotly: Temperature Impact on Yield
        st.markdown("**Temperature Stress Impact**")
        fig_temp = px.scatter(sample_df, x='temperature', y='milk_yield', color='humidity',
                              title="Heat Stress Threshold Visualization")
        # Add marker for simulated cow
        fig_temp.add_trace(go.Scatter(x=[sim_temp], y=[predicted_yield], mode='markers',
                                      marker=dict(color='yellow', size=15, line=dict(color='black', width=2)),
                                      name='Simulated Cow'))
        st.plotly_chart(fig_temp, use_container_width=True)
        
else:
    st.info("System is waiting for generated data and trained models. Execute setup scripts first.")
