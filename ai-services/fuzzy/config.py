"""Seluruh parameter Fuzzy Mamdani ada di file ini: himpunan fuzzy, fungsi keanggotaan, dan aturan.

PENTING: nilai di bawah adalah USULAN AWAL yang masuk akal, BELUM dikalibrasi dengan data pasar.
Kalibrasi dan validasi dilakukan lewat evaluate.py dengan data acuan (lihat README).
Mengubah angka di sini tidak memerlukan perubahan kode lain.

Format himpunan: ("tri", [a, b, c]) segitiga  |  ("trap", [a, b, c, d]) trapesium.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"
VERSION = "fuzzy-1.0.0"
ROUND_TO = 10_000            # harga rekomendasi dibulatkan ke Rp10.000 terdekat

# ------------------------------------------------------------ INPUT 1: kondisi fisik (dari CNN)
# skor_kondisi: 0-100, 100 = kondisi terbaik.
KONDISI_UNIVERSE = (0.0, 100.0)
KONDISI_SETS = {
    "Buruk":      ("trap", [0, 0, 40, 55]),
    "Sedang":     ("tri",  [40, 62, 80]),
    "Baik":       ("tri",  [65, 82, 95]),
    "SangatBaik": ("trap", [85, 95, 100, 100]),
}

# ------------------------------------------------------------ INPUT 2: estimasi harga pasar (dari Random Forest)
# Satuan Rupiah. Sesuaikan dengan sebaran harga di dataset Anda (mis. berdasarkan kuartil resale_price).
HARGA_UNIVERSE = (0.0, 40_000_000.0)
HARGA_SETS = {
    "Murah":    ("trap", [0, 0, 3_000_000, 6_000_000]),
    "Menengah": ("tri",  [3_000_000, 8_000_000, 15_000_000]),
    "Mahal":    ("trap", [10_000_000, 18_000_000, 40_000_000, 40_000_000]),
}

# ------------------------------------------------------------ OUTPUT: rasio harga rekomendasi (% dari estimasi pasar)
# harga_rekomendasi = estimasi_harga_pasar x rasio / 100
RASIO_UNIVERSE = (30.0, 100.0)
# Segitiga SIMETRIS (kaki kiri = kaki kanan): saat hanya satu aturan aktif, titik pusat tidak bergeser
# ketika kekuatan aturan berubah, sehingga keluaran naik-turun secara wajar terhadap input.
RASIO_SETS = {
    "SangatRendah": ("tri",  [30, 44, 58]),    # titik pusat ~44%
    "Rendah":       ("tri",  [44, 58, 72]),    # ~58%
    "Sedang":       ("tri",  [58, 72, 86]),    # ~72%
    "Tinggi":       ("tri",  [72, 86, 100]),   # ~86%
    "SangatTinggi": ("trap", [86, 95, 100, 100]),   # ~95%
}
GRID_STEP = 0.1              # resolusi defuzzifikasi (centroid)

# ------------------------------------------------------------ ATURAN IF-THEN
# (kondisi, harga, rasio)  ->  IF kondisi=... AND harga=... THEN rasio=...
# Hipotesis awal yang HARUS divalidasi: saat kondisi baik, harga tidak memengaruhi rasio;
# saat kondisi buruk, perangkat yang lebih mahal mempertahankan nilai relatif lebih tinggi.
RULES = [
    ("SangatBaik", "Murah",    "SangatTinggi"),
    ("SangatBaik", "Menengah", "SangatTinggi"),
    ("SangatBaik", "Mahal",    "SangatTinggi"),
    ("Baik",       "Murah",    "Tinggi"),
    ("Baik",       "Menengah", "Tinggi"),
    ("Baik",       "Mahal",    "Tinggi"),
    ("Sedang",     "Murah",    "Rendah"),
    ("Sedang",     "Menengah", "Sedang"),
    ("Sedang",     "Mahal",    "Sedang"),
    ("Buruk",      "Murah",    "SangatRendah"),
    ("Buruk",      "Menengah", "Rendah"),
    ("Buruk",      "Mahal",    "Rendah"),
]
