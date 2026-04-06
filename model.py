import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

print("Training AI model...")

# Small training data (this teaches the AI)
data = {
    'people': [2, 3, 4, 5, 4, 6, 3, 5],
    'ac_hours': [4, 6, 8, 10, 6, 12, 5, 9],
    'bill': [1500, 2000, 2500, 3000, 2200, 3500, 1800, 2800],
    'energy_kwh': [250, 320, 420, 550, 380, 680, 290, 480],
    'co2_kg': [180, 230, 300, 390, 270, 490, 210, 340]
}

df = pd.DataFrame(data)

# Train two simple AI models
energy_model = LinearRegression()
energy_model.fit(df[['people', 'ac_hours', 'bill']], df['energy_kwh'])

co2_model = LinearRegression()
co2_model.fit(df[['people', 'ac_hours', 'bill']], df['co2_kg'])

# Save the trained models
joblib.dump(energy_model, 'energy_model.joblib')
joblib.dump(co2_model, 'co2_model.joblib')

print("✅ AI Models trained and saved successfully!")