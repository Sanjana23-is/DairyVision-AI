"""
data_generation.py
Generates a realistic synthetic dataset for dairy farm simulation.
"""
import pandas as pd
import numpy as np
import uuid
import os
from utils import calculate_thi, calculate_lactation_multiplier, calculate_heat_stress_penalty

np.random.seed(42)

def generate_dairy_data(num_records: int = 30000) -> pd.DataFrame:
    """
    Generate synthetic dairy farm data.
    Features: cow_id, age, breed, feed_quantity, temperature, humidity, lactation_stage, milk_yield
    """
    print(f"Generating {num_records} records of synthetic dairy data...")
    
    breeds = ['Holstein', 'Jersey', 'Guernsey']
    breed_base_yield = {'Holstein': 32.0, 'Jersey': 22.0, 'Guernsey': 20.0}
    
    data = []
    
    # Generate 1000 unique cows
    num_cows = 1000
    cows = []
    for _ in range(num_cows):
        cows.append({
            'cow_id': str(uuid.uuid4())[:8],
            'breed': np.random.choice(breeds, p=[0.6, 0.3, 0.1]),
            'age': np.random.randint(2, 11)  # age between 2 and 10
        })
        
    for _ in range(num_records):
        cow = cows[np.random.randint(0, num_cows)]
        
        # Environmental and state variables
        temperature = np.random.normal(loc=20.0, scale=10.0)  # -10C to 50C
        temperature = np.clip(temperature, -10, 45)
        humidity = np.random.normal(loc=60.0, scale=15.0)
        humidity = np.clip(humidity, 20, 100)
        
        lactation_stage = np.random.randint(1, 400) # Days in milk
        
        # Feed quantity (kg/day)
        # Assuming average 24kg for Holstein, less for others
        base_feed = 24.0 if cow['breed'] == 'Holstein' else 18.0
        feed_quantity = np.random.normal(loc=base_feed, scale=3.0)
        feed_quantity = np.clip(feed_quantity, 10.0, 35.0)
        
        # --- Target Variable Logic (milk_yield) ---
        base_yield = breed_base_yield[cow['breed']]
        
        # Effect of age: peaks around 4-6 years
        age_multiplier = 1.0 - (abs(cow['age'] - 5) * 0.05)
        
        # Effect of lactation stage
        lac_multiplier = calculate_lactation_multiplier(lactation_stage)
        
        # Effect of environment (Heat stress)
        thi = calculate_thi(temperature, humidity)
        stress_penalty = calculate_heat_stress_penalty(thi)
        
        # Effect of feed: Logarithmic returns
        # Optimal feed roughly equal to base_feed. Less feed = less milk
        feed_multiplier = np.log1p(feed_quantity) / np.log1p(base_feed)
        
        # Calculate raw yield
        expected_yield = base_yield * age_multiplier * lac_multiplier * stress_penalty * feed_multiplier
        
        # Add random noise for realism
        noise = np.random.normal(0, 1.5)
        final_yield = max(0.0, expected_yield + noise)
        
        data.append({
            'cow_id': cow['cow_id'],
            'age': cow['age'],
            'breed': cow['breed'],
            'feed_quantity': round(feed_quantity, 2),
            'temperature': round(temperature, 2),
            'humidity': round(humidity, 2),
            'lactation_stage': lactation_stage,
            'milk_yield': round(final_yield, 2)
        })
        
    df = pd.DataFrame(data)
    
    # Save to CSV in the current folder
    output_file = 'dairy_data.csv'
    df.to_csv(output_file, index=False)
    print(f"Data successfully generated and saved to {output_file}")
    
    return df

if __name__ == "__main__":
    generate_dairy_data()
