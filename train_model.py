# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 🔹 1. Veriyi yükle
data = pd.read_csv("data/merged_data.csv")

# 🔹 2. Gerekli sütunların olup olmadığını kontrol et
required_columns = ['hdi', 'gdp_per_capita', 'edu_expenditure_pct', 'Tuition_USD', 'Living_Cost_Index']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"❌ Gerekli sütun eksik: {col}")

# 🔹 3. Eksik verileri temizle
data = data.dropna(subset=required_columns)

# 🔹 4. Özellikleri (X) ve hedef değişkeni (y) ayır
features = ['gdp_per_capita', 'edu_expenditure_pct', 'Tuition_USD', 'Living_Cost_Index']
X = data[features]
y = data['hdi']

# 🔹 5. Eğitim/test verisi olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 6. Modeli kur ve eğit
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🔹 7. Test verisiyle tahmin yap
y_pred = model.predict(X_test)

# 🔹 8. Başarı metriklerini hesapla
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n📈 RMSE: {rmse:.4f}")
print(f"📊 R² Score: {r2:.4f}")

# 🔹 9. Modeli dosyaya kaydet
joblib.dump(model, "hdi_model.pkl")
print("\n✅ Model başarıyla kaydedildi: hdi_model.pkl")
