"""Konfigurasi pusat. Kalau skema dataset berubah, cukup ubah file ini."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "dataset_harga.csv"   # boleh juga .xlsx
MODEL_PATH = BASE_DIR / "models" / "rf_price_model.joblib"
META_PATH = BASE_DIR / "models" / "model_meta.json"
REPORT_DIR = BASE_DIR / "reports"

MODEL_VERSION = "rf-1.0.0"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---------------------------------------------------------------- skema data
TARGET = "resale_price"

# Nama kolom alternatif -> nama baku (dibaca otomatis saat memuat data)
COLUMN_ALIASES = {
    "harga_jual": "resale_price", "harga_bekas": "resale_price",
    "merek": "brand", "tahun_rilis": "release_year",
    "ram_gb": "ram", "storage_gb": "storage",
    "harga_launching": "original_price", "harga_awal": "original_price",
    "kapasitas_baterai": "battery_capacity", "processor": "processor_name",
}

# Kolom yang tidak dipakai model (hanya informasi).
DROP_COLUMNS = ["network"]

# Kandidat fitur. Yang dipakai model = kandidat yang ADA di dataset dan
# bervariasi (kolom konstan, misalnya brand saat hanya iPhone, otomatis dilewati).
NUMERIC_CANDIDATES = [
    "release_year", "ram", "storage", "battery_capacity", "original_price",
    "usia_pakai_bulan", "kesehatan_baterai",
]
CATEGORICAL_CANDIDATES = ["brand", "model", "processor_name", "kelengkapan", "garansi"]

# Wajib ada di dataset (dan wajib terisi per baris).
REQUIRED_COLUMNS = ["brand", "model", "storage", "release_year", "ram", "original_price"]

# Kolom spesifikasi bawaan model: pengguna cukup memilih brand, model, dan storage;
# nilai kolom ini diambil dari katalog data latih.
SPEC_NUMERIC = ["release_year", "ram", "battery_capacity", "original_price"]
SPEC_CATEGORICAL = ["processor_name"]

# Fitur kondisi/pemakaian unit (diisi dari listing nyata, opsional).
USAGE_FEATURES = ["usia_pakai_bulan", "kesehatan_baterai", "kelengkapan", "garansi"]

# Kolom opsional untuk mencegah data leakage: ID penjual/unit yang sama tidak
# boleh terbagi ke data latih dan uji.
GROUP_COL = "id_penjual"
