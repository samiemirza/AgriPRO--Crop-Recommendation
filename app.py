from flask import Flask, request, jsonify, render_template
import os
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# ✅ OpenRouter API Key (for DeepSeek routing)
OPENROUTER_API_KEY = "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"

# Load dataset
df = pd.read_csv("crop_yield_dataset.csv")
df = df[df["Crop_Yield"] > 0]

# Binning functions
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

# Apply binning and train model
df['Temperature_Bin'] = df['Temperature'].apply(bin_temperature)
df['Humidity_Bin'] = df['Humidity'].apply(bin_humidity)
df['Soil_pH_Bin'] = df['Soil_pH'].apply(bin_ph)
df['Soil_Quality_Bin'] = df['Soil_Quality'].apply(bin_soil_quality)
df = pd.get_dummies(df, columns=['Crop_Type'])

features = ['Soil_Type', 'Temperature_Bin', 'Humidity_Bin', 'Soil_pH_Bin', 'Soil_Quality_Bin', 'Wind_Speed', 'N', 'P', 'K']
target_cols = [col for col in df.columns if col.startswith('Crop_Type_')]

X = df[features]
y = df[target_cols]

categorical_features = ['Soil_Type', 'Temperature_Bin', 'Humidity_Bin', 'Soil_pH_Bin', 'Soil_Quality_Bin']
numeric_features = ['Wind_Speed', 'N', 'P', 'K']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', 'passthrough', numeric_features)
])

model = Pipeline([
    ('prep', preprocessor),
    ('multi_gbr', MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    ))
])
model.fit(X, y)

# Prediction logic
def get_top_crops(input_data):
    input_df = pd.DataFrame([input_data])
    input_df['Temperature_Bin'] = input_df['Temperature'].apply(bin_temperature)
    input_df['Humidity_Bin'] = input_df['Humidity'].apply(bin_humidity)
    input_df['Soil_pH_Bin'] = input_df['Soil_pH'].apply(bin_ph)
    input_df['Soil_Quality_Bin'] = input_df['Soil_Quality'].apply(bin_soil_quality)
    processed_input = input_df[features]
    predicted = model.predict(processed_input)[0]
    results = dict(zip(target_cols, predicted))
    top3 = sorted(results.items(), key=lambda x: x[1], reverse=True)[:3]
    return [(k.replace("Crop_Type_", ""), round(v, 2)) for k, v in top3]

# DeepSeek prompt
def generate_prompt(input_data, top_crops):
    summary = (
        f"Temperature = {bin_temperature(input_data['Temperature']).replace('_', ' ').title()}, "
        f"Humidity = {bin_humidity(input_data['Humidity']).replace('_', ' ').title()}, "
        f"Soil pH = {bin_ph(input_data['Soil_pH']).replace('_', ' ').title()}, "
        f"Soil Quality = {bin_soil_quality(input_data['Soil_Quality']).replace('_', ' ').title()}, "
        f"Soil Type = {input_data['Soil_Type']}, "
        f"N = {input_data['N']}, P = {input_data['P']}, K = {input_data['K']}, Wind Speed = {input_data['Wind_Speed']}."
    )
    crops = ", ".join([name for name, _ in top_crops])
    return f"Based on the given conditions: {summary} Why are {crops} optimal crop choices for this environment?"

def query_openrouter(prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek/deepseek-r1:free",
        "messages": [
            {"role": "system", "content": "You are a helpful agricultural expert."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }

    print("🔼 Prompt being sent to OpenRouter:")
    print(prompt)
    print("📦 Payload being sent:")
    print(payload)

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)

    print("🔽 Status code:", response.status_code)
    print("🧾 Raw response text:", response.text)

    if response.status_code == 200:
        try:
            content = response.json()['choices'][0]['message']['content']
            print("✅ Parsed content from response:", content)
            return content
        except Exception as e:
            print("❌ Error parsing JSON content:", e)
            print("🧾 Full response JSON:", response.json())
            return "⚠️ Could not parse AI reply properly."
    else:
        raise Exception(f"OpenRouter API Error: {response.status_code} - {response.text}")


# Routes
@app.route("/")
def home():
    return render_template("home.html")  # ⬅ Default route now points to home

@app.route("/predict")
def predict_page():
    return render_template("index.html")  # ⬅ Your prediction tool UI

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.json
        top_crops = get_top_crops(input_data)
        prompt = generate_prompt(input_data, top_crops)
        explanation = query_openrouter(prompt)
        return jsonify({
            "top_crops": top_crops,
            "prompt": prompt,
            "deepseek_explanation": explanation
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

