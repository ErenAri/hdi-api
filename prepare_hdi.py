import pandas as pd
import unicodedata

def normalize_country(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
    return name.replace("ü", "u").replace("ş", "s").replace("ö", "o") \
               .replace("ç", "c").replace("ğ", "g").replace("ı", "i")

file_path = "data/undp_hdi.xlsx"
df = pd.read_excel(file_path, sheet_name=0, header=4)

# doğru başlıklar: Country + yıllar (ör: 1990–2023)
year_cols = [col for col in df.columns if str(col).isdigit()]
df = df[["Country"] + year_cols]

# ülke isimlerini normalize et
df["country"] = df["Country"].astype(str).apply(normalize_country)
df = df.drop(columns=["Country"])

# uzun forma çevir
df = df.melt(id_vars="country", var_name="year", value_name="hdi")
df["year"] = df["year"].astype(int)
df["hdi"] = pd.to_numeric(df["hdi"], errors="coerce")
df = df.dropna()

# CSV'ye kaydet
df.to_csv("data/hdi_clean.csv", index=False, encoding="utf-8-sig")
print("✅ hdi_clean.csv başarıyla oluşturuldu.")
