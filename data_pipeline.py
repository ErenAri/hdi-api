# data_pipeline.py

import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

def download_kaggle_data(dataset='adilshamim8/cost-of-international-education', download_path='data/'):
    """Kaggle veri setini indirir ve açar"""
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=download_path, unzip=True)
    print(f"Kaggle veri seti '{dataset}' indirildi ve '{download_path}' klasörüne açıldı.")

def load_kaggle_data(path="data/international_education_costs.csv"):
    df = pd.read_csv(path)
    print("📄 Kaggle sütunları:", df.columns.tolist())

    # Sütun adını normalize et
    if 'Country' in df.columns:
        df.rename(columns={'Country': 'country'}, inplace=True)
    elif 'Location' in df.columns:
        df.rename(columns={'Location': 'country'}, inplace=True)

    return df


def load_undp_hdi(path="data/undp_hdi.xlsx"):
    """UNDP HDI verisini oku ve uzun forma çevir"""
    df = pd.read_excel(path, sheet_name=0, header=4)
    print("📋 Gerçek Sütunlar:", df.columns.tolist())

    # Sayı olan yılları ayıkla
    year_cols = [col for col in df.columns if str(col).isdigit()]
    df = df[["Country"] + year_cols]
    df = df.melt(id_vars="Country", var_name="year", value_name="hdi")
    df.columns = ['country', 'year', 'hdi']
    df['year'] = df['year'].astype(int)
    df['hdi'] = pd.to_numeric(df['hdi'], errors='coerce')
    return df

def load_worldbank_manual_data():
    """World Bank GDP & Eğitim Harcaması CSV verilerini oku"""
    gdp_df = pd.read_csv("data/gdp_per_capita.csv", skiprows=4)
    edu_df = pd.read_csv("data/edu_expenditure_pct.csv", skiprows=4)

    gdp = gdp_df.melt(id_vars="Country Name", var_name="year", value_name="gdp_per_capita")
    edu = edu_df.melt(id_vars="Country Name", var_name="year", value_name="edu_expenditure_pct")

    gdp['year'] = pd.to_numeric(gdp['year'], errors='coerce')
    edu['year'] = pd.to_numeric(edu['year'], errors='coerce')

    gdp.rename(columns={'Country Name': 'country'}, inplace=True)
    edu.rename(columns={'Country Name': 'country'}, inplace=True)

    merged = pd.merge(gdp, edu, on=['country', 'year'], how='outer')
    return merged

def merge_datasets(ed_df, hdi_df, wb_df=None):
    """Tüm verileri birleştir"""
    df = hdi_df
    if wb_df is not None:
        df = df.merge(wb_df, on=['country', 'year'], how='left')
    if ed_df is not None:
        # Kaggle verisi ülke bazlı, yıl içermediği için sadece ülkeye göre merge edilir
        df = df.merge(ed_df, on='country', how='left')
    return df

if __name__ == "__main__":
    # 1. Kaggle veri setini indir
    download_kaggle_data()

    # 2. Kaggle CSV dosyasını yükle
    ed = load_kaggle_data()

    # 3. UNDP HDI verisini yükle
    hdi = load_undp_hdi()

    # 4. World Bank verilerini manuel yükle
    wb = load_worldbank_manual_data()

    # 5. Birleştir
    data = merge_datasets(ed, hdi, wb)

    # 6. Ön izleme
    print("\n📊 Birleştirilmiş Veri:")
    print(data.head(10))

    # 7. Veriyi diske kaydet
    data.to_csv("data/merged_data.csv", index=False, encoding="utf-8-sig")
    print("📁 merged_data.csv dosyası kaydedildi.")


