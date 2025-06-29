from flask import Flask, request, jsonify, render_template
import joblib
import json
import requests
import numpy as np

# --- DeepSeek/OpenRouter API Key (keep secure!) ---
OPENROUTER_API_KEY = "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"

app = Flask(__name__)

# --- Load Models, Encoders, Feature List at Startup ---
clf = joblib.load('model_classifier.pkl')
reg = joblib.load('model_regressor.pkl')
encoders = joblib.load('encoder.pkl')
with open('features.json') as f:
    FEATURES = json.load(f)

# --- Crop Months Lookup (for rotation logic) ---
CROP_MONTHS = {
    'wheat':      (11, 4),
    'corn':       (2, 7),
    'rice':       (6, 10),
    'barley':     (10, 3),
    'soybean':    (6, 10),
    'cotton':     (4, 10),
    'sugarcane':  (2, 11),
    'tomato':     (8, 11),
    'potato':     (10, 1),
    'sunflower':  (1, 5)
}

# --- Preprocess incoming data for model ---
def preprocess_input(input_dict):
    # Follow order in features.json!
    row = []
    for feat in FEATURES:
        if feat == 'Soil_Type':
            val = encoders['soil_type'].transform([input_dict['Soil_Type'].strip()])[0]
        elif feat == 'Soil_Quality_Class':
            val = encoders['soil_quality'].transform([input_dict['Soil_Quality_Class'].strip().lower()])[0]
        else:
            val = float(input_dict[feat])
        row.append(val)
    return [row]

# --- Predict Top Crops and their Expected Yields ---
def recommend_crops(input_dict, top_n=3):
    X = preprocess_input(input_dict)
    # Predict probabilities for each crop (classifier), yields for each (regressor)
    class_probs = clf.predict_proba(X)[0]
    # Handle if only one class (rare, edge case)
    if class_probs.ndim == 0:
        class_probs = np.array([1.0])
    # Crop labels and names
    crop_labels = list(range(len(class_probs)))
    crop_names = encoders['crop_type'].inverse_transform(crop_labels)
    # Regression (yield prediction for each crop, one-by-one)
    yields = []
    for i, crop_label in enumerate(crop_labels):
        X_crop = X.copy()
        X_crop = np.array(X_crop)
        # Swap in this crop as label for prediction, or you may need a custom logic if your regressor is multi-output
        # (If regressor predicts yield for a given input/crop)
        yields.append(reg.predict(X)[0])  # Placeholder for generic regressor, else adjust as needed

    # Rank crops by classifier probability (not yield! unless you want by yield)
    crops_ranked = sorted(zip(crop_names, class_probs, yields), key=lambda x: x[1], reverse=True)
    recommendations = []
    for i, (crop, prob, yld) in enumerate(crops_ranked[:top_n]):
        recommendations.append({
            "crop": crop,
            "expected_yield_maund_per_acre": round(yld, 2),
            "probability": round(prob, 3)
        })
    return recommendations

# --- Explainability: DeepSeek Prompt Logic ---
def generate_prompt(input_data, recommendations):
    summary = (
        f"Temperature = {input_data['Temperature']}, "
        f"Humidity = {input_data['Humidity']}, "
        f"Soil pH = {input_data['Soil_pH']}, "
        f"Soil Quality = {input_data['Soil_Quality']}, "
        f"Soil Type = {input_data['Soil_Type']}, "
        f"N = {input_data['N']}, P = {input_data['P']}, K = {input_data['K']}, Wind Speed = {input_data['Wind_Speed']}."
    )
    crops = ", ".join([rec['crop'] for rec in recommendations])
    return f"Based on the given conditions: {summary} Why are {crops} optimal crop choices for this environment in Pakistan?"

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
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        try:
            content = response.json()['choices'][0]['message']['content']
            return content
        except Exception:
            return "⚠️ Could not parse AI reply properly."
    else:
        return f"API Error: {response.status_code}"

# --- Crop Rotation Recommendation ---
# Basic cereal-legume-others logic
CROP_GROUPS = {
    'wheat': 'cereal',
    'rice': 'cereal',
    'barley': 'cereal',
    'corn': 'cereal',
    'soybean': 'legume',
    'cotton': 'other',
    'sugarcane': 'other',
    'tomato': 'other',
    'potato': 'other',
    'sunflower': 'other',
}

def recommend_rotation(top_crop):
    this_group = CROP_GROUPS.get(top_crop, 'other')
    # Prefer different group, else next in list
    for crop, group in CROP_GROUPS.items():
        if crop != top_crop and group != this_group:
            sow, harvest = CROP_MONTHS[crop]
            return {
                "next_crop": crop,
                "sowing_month": sow,
                "harvesting_month": harvest
            }
    # Fallback: any different crop
    for crop in CROP_MONTHS:
        if crop != top_crop:
            sow, harvest = CROP_MONTHS[crop]
            return {
                "next_crop": crop,
                "sowing_month": sow,
                "harvesting_month": harvest
            }
    return {}


# --- Flask Routes ---
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict_page():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        input_data = request.json
        recommendations = recommend_crops(input_data, top_n=3)
        prompt = generate_prompt(input_data, recommendations)
        deepseek_explanation = query_openrouter(prompt)
        # Crop rotation based on the 1st recommended crop
        rotation = recommend_rotation(recommendations[0]["crop"])
        return jsonify({
            "recommendations": recommendations,
            "deepseek_explanation": deepseek_explanation,
            "rotation_plan": rotation
        })
    except Exception as e:
        print("Error in /recommend:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
