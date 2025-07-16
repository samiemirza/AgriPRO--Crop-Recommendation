import os
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
import joblib
import json
import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
app = Flask(__name__)

# --- Load Models, Encoders, Feature List at Startup ---
clf = joblib.load('model_classifier.pkl')
reg = joblib.load('model_regressor.pkl')
encoders = joblib.load('encoder.pkl')
with open('features.json') as f:
    FEATURES = json.load(f)

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

def preprocess_input(input_dict):
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

def recommend_crops(input_dict, top_n=3):
    X = preprocess_input(input_dict)
    class_probs = clf.predict_proba(X)[0]
    if class_probs.ndim == 0:
        class_probs = np.array([1.0])
    crop_labels = list(range(len(class_probs)))
    crop_names = encoders['crop_type'].inverse_transform(crop_labels)
    yields = []
    for i, crop_label in enumerate(crop_labels):
        X_crop = X.copy()
        X_crop = np.array(X_crop)
        yields.append(reg.predict(X)[0])  # If you have per-crop regression, adjust logic here
    crops_ranked = sorted(zip(crop_names, class_probs, yields), key=lambda x: x[1], reverse=True)
    recommendations = []
    for i, (crop, prob, yld) in enumerate(crops_ranked[:top_n]):
        recommendations.append({
            "crop": crop,
            "expected_yield_maund_per_acre": round(yld, 2),
            "probability": round(prob, 3)
        })
    return recommendations

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

# --- Streaming DeepSeek via OpenRouter ---
def stream_openrouter(prompt):
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
        "max_tokens": 2000,
        "stream": True
    }
    with requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        stream=True,
        timeout=60
    ) as r:
        for line in r.iter_lines():
            if line and line.startswith(b"data: "):
                chunk = line[len(b"data: "):].decode("utf-8")
                if chunk.strip() == "[DONE]":
                    break
                try:
                    # For OpenAI-style response
                    content = json.loads(chunk)["choices"][0]["delta"].get("content", "")
                    if content:
                        yield content
                except Exception:
                    continue

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
    for crop, group in CROP_GROUPS.items():
        if crop != top_crop and group != this_group:
            sow, harvest = CROP_MONTHS[crop]
            return {
                "next_crop": crop,
                "sowing_month": sow,
                "harvesting_month": harvest
            }
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
        rotation = recommend_rotation(recommendations[0]["crop"])
        return jsonify({
            "recommendations": recommendations,
            "explanation_prompt": prompt,
            "rotation_plan": rotation
        })
    except Exception as e:
        print("Error in /recommend:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/explanation_stream", methods=["POST"])
def explanation_stream():
    try:
        data = request.json
        if not data:
            return Response("Error: No JSON data provided", mimetype='text/plain')
        prompt = data.get("prompt")
        if not prompt:
            return Response("Error: No prompt provided", mimetype='text/plain')
        def generate():
            for chunk in stream_openrouter(prompt):
                yield chunk
        return Response(stream_with_context(generate()), mimetype='text/plain')
    except Exception as e:
        print("Error in /explanation_stream:", e)
        return Response("Error: " + str(e), mimetype='text/plain')

if __name__ == "__main__":
    app.run(debug=True)

