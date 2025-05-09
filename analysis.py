# analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veriyi yükle
data = pd.read_csv("data/merged_data.csv")  # Eğer kayıt etmediysen direkt data_pipeline.py'den aktarabilirim

# Örnek: HDI'nin yıllara göre değişimi (Norveç, Türkiye, Hindistan)
countries = ['Norway', 'Turkey', 'India']
for c in countries:
    subset = data[data['country'] == c]
    plt.plot(subset['year'], subset['hdi'], label=c)

plt.title("HDI Trendleri")
plt.xlabel("Yıl")
plt.ylabel("HDI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
