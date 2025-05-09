from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
import numpy as np
import unicodedata

app = Flask(__name__)
CORS(app)

def normalize_country(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
    return name.replace("ü", "u").replace("ş", "s").replace("ö", "o") \
               .replace("ç", "c").replace("ğ", "g").replace("ı", "i")

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
        country_data = df[df["country"] == country_norm]

        if len(country_data) < 5:
            return jsonify({"error": "Yeterli HDI verisi yok."}), 404

        X = country_data["year"].values.reshape(-1, 1)
        y = country_data["hdi"].values
        model = LinearRegression()
        model.fit(X, y)

        base_year = int(country_data["year"].max())
        future_years = list(range(base_year + 1, base_year + 51))

        historical = [
            {"year": int(row["year"]), "hdi": float(row["hdi"]), "predicted": False}
            for _, row in country_data.iterrows()
        ]

        future = []
        for year in future_years:
            pred = float(model.predict([[year]]))
            pred = max(0.0, min(pred, 1.0))
            future.append({"year": year, "hdi": round(pred, 4), "predicted": True})

        full_data = historical + future
        full_data_sorted = sorted(full_data, key=lambda x: x["year"])

        return jsonify({
            "country": country_input,
            "data": full_data_sorted
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
