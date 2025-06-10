# train_model.py
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

#  ─── Your binning functions ──────────────────────────────────────────────
def bin_temperature(temp):
    if temp <= 10: return 'very_low'
    elif temp <= 20: return 'low'
    elif temp <= 30: return 'moderate'
    elif temp <= 35: return 'high'
    return 'very_high'

def bin_humidity(h):
    if h <= 30: return 'very_low'
    elif h <= 50: return 'low'
    elif h <= 70: return 'moderate'
    elif h <= 85: return 'high'
    return 'very_high'

def bin_ph(ph):
    if ph <= 5.5: return 'very_low'
    elif ph <= 6.5: return 'low'
    elif ph <= 7.5: return 'moderate'
    elif ph <= 8.5: return 'high'
    return 'very_high'

def bin_soil_quality(sq):
    if sq <= 2: return 'very_low'
    elif sq <= 4: return 'low'
    elif sq <= 6: return 'moderate'
    elif sq <= 8: return 'high'
    return 'very_high'
# ─────────────────────────────────────────────────────────────────────────

# 1. Load and preprocess
df = pd.read_csv("crop_yield_dataset.csv")
df = df[df["Crop_Yield"] > 0]
df['Temperature_Bin']   = df['Temperature'].apply(bin_temperature)
df['Humidity_Bin']      = df['Humidity'].apply(bin_humidity)
df['Soil_pH_Bin']       = df['Soil_pH'].apply(bin_ph)
df['Soil_Quality_Bin']   = df['Soil_Quality'].apply(bin_soil_quality)
df = pd.get_dummies(df, columns=['Crop_Type'])

# 2. Define features/targets
features    = ['Soil_Type','Temperature_Bin','Humidity_Bin','Soil_pH_Bin','Soil_Quality_Bin','Wind_Speed','N','P','K']
target_cols = [c for c in df.columns if c.startswith('Crop_Type_')]
X, y        = df[features], df[target_cols]

# 3. Build & fit pipeline
cat_feats = ['Soil_Type','Temperature_Bin','Humidity_Bin','Soil_pH_Bin','Soil_Quality_Bin']
num_feats = ['Wind_Speed','N','P','K']
pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats),
                         ('num', 'passthrough',                 num_feats)])
pipe = Pipeline([
    ('prep',    pre),
    ('predict', MultiOutputRegressor(GradientBoostingRegressor(
                   n_estimators=100, learning_rate=0.1, random_state=42)))
])
pipe.fit(X, y)

# 4. Save the fitted pipeline
os.makedirs("models", exist_ok=True)
joblib.dump(pipe, "models/crop_recommender.joblib")

print("✅ Model trained and saved to models/crop_recommender.joblib")


