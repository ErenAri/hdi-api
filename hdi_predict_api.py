from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
import numpy as np
import unicodedata

app = Flask(__name__)
CORS(app)

# 📥 Ülke normalizasyonu
def normalize_country(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
    return name.replace("ü", "u").replace("ş", "s").replace("ö", "o") \
               .replace("ç", "c").replace("ğ", "g").replace("ı", "i")

# 📊 HDI verisini yükle
df = pd.read_csv("data/hdi_clean.csv", encoding="utf-8-sig")
df["country"] = df["country"].astype(str).apply(normalize_country)

@app.route("/")
def home():
    return "✅ HDI tahmin API çalışıyor."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        country_input = data.get("country")
        if not country_input:
            return jsonify({"error": "Ülke bilgisi eksik."}), 400

        country_norm = normalize_country(country_input)

        # ✅ Alias eşlemesi
        aliases = {
            "turkey": "turkiye",
            "usa": "united states",
            "uk": "united kingdom",
            "south korea": "korea (republic of)",
            "north korea": "korea (democratic people's republic of)"
        }
        country_norm = aliases.get(country_norm, country_norm)

        # Veri çek
        country_data = df[df["country"] == country_norm]

        # 🔍 Debug çıktısı
        print("📥 Gelen ülke:", country_input)
        print("🔎 Normalize edilmiş:", country_norm)
        print("🧾 Eşleşen veri satırı sayısı:", len(country_data))
        print(country_data.head())

        if len(country_data) < 5:
            return jsonify({"error": "Yeterli HDI verisi yok."}), 404

        # Regressyon modeli
        X = country_data["year"].values.reshape(-1, 1)
        y = country_data["hdi"].values
        model = LinearRegression()
        model.fit(X, y)

        base_year = country_data["year"].max()
        predictions = {}
        for delta in [10, 30, 50]:
            future_year = base_year + delta
            pred = float(model.predict([[future_year]]))
            pred = max(0.0, min(pred, 1.0))
            predictions[f"year_{future_year}"] = round(pred, 4)

        return jsonify({
            "country": country_input,
            "base_year": int(base_year),
            "predictions": predictions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
