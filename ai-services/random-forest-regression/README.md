# Random Forest Regression – Prediksi Harga Pasar Smartphone Bekas

Layanan AI untuk Penelitian 1. Berjalan sebagai API FastAPI (port **8001**) yang dipanggil Laravel.

## Struktur
```
random-forest-regression/
├── config.py          # skema kolom, path, parameter (ubah di sini jika dataset berubah)
├── data_utils.py      # pembacaan & pembersihan data, pembuatan katalog
├── train.py           # training + evaluasi + simpan model
├── main.py            # API: /health, /catalog, /predict
├── dataset/           # dataset_harga.csv (atau .xlsx) + template.csv
├── models/            # rf_price_model.joblib, model_meta.json (hasil train.py)
├── reports/           # grafik evaluasi untuk laporan
└── notebooks/         # eksplorasi data (EDA)
```

## Setup (Windows)
```cmd
cd ai-services\random-forest-regression
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Alur kerja
```cmd
python train.py                    # latih dari dataset\dataset_harga.csv
uvicorn main:app --reload --port 8001
```
Opsi training: `--data file.xlsx`, `--n-iter 40` (tuning lebih banyak), `--no-plots`.
Setiap kali data berubah: jalankan `train.py` lagi, lalu restart uvicorn.

## Dataset
Taruh di `dataset/dataset_harga.csv` (CSV UTF-8 atau Excel).

| Kolom | Status | Keterangan |
|---|---|---|
| `brand`, `model`, `storage`, `release_year`, `ram`, `original_price` | **Wajib** | Identitas dan spesifikasi varian |
| `resale_price` | **Wajib (target)** | Harga bekas (Rp) |
| `processor_name`, `battery_capacity` | Opsional | Spesifikasi tambahan |
| `usia_pakai_bulan`, `kesehatan_baterai`, `kelengkapan`, `garansi` | Opsional, **sangat disarankan** | Kondisi unit dari listing nyata. Otomatis dipakai model jika ada |
| `id_penjual` | Opsional | Mencegah data leakage: penjual yang sama tidak terbagi ke data latih dan uji |
| `sumber`, `tanggal_pengumpulan` | Opsional | Dokumentasi asal data (tidak dipakai model) |
| `network` | Diabaikan | |

Otomatis ditangani: angka berformat `43,999,000` / `Rp 5.500.000` / `98%`, penulisan `Iphone`/`iphone` → `iPhone`, baris duplikat, harga ≤ 0, kolom yang nilainya konstan (misalnya `brand` saat hanya iPhone). Nama kolom Indonesia (`harga_jual`, `merek`, dst.) juga dikenali (lihat `COLUMN_ALIASES`).

**Menambah merek lain atau fitur kondisi nanti:** cukup tambahkan baris/kolom ke dataset lalu jalankan `train.py` ulang. Tidak perlu mengubah kode.

## Membaca hasil training
- **Holdout**: dibandingkan dengan Baseline dan Linear Regression.
- **Cross-validation**: lebih stabil untuk data kecil.
- **`PERINGATAN`** di output ikut tersimpan di `models/model_meta.json`. Selama ada peringatan "satu varian satu harga", jangan memakai skor evaluasi untuk laporan.

## Kontrak API

### `GET /health`
Status server, versi model, fitur yang dipakai, dan peringatan model.

### `GET /catalog`
Daftar brand, model, dan storage yang dikenal. Dipakai Laravel untuk mengisi dropdown.
```json
{"brands":[{"brand":"Apple","models":[{"model":"iPhone 15 Pro","storages":[128,256,512,1024]}]}]}
```

### `POST /predict`
Wajib: `brand`, `model`, `storage`. Field lain opsional; spesifikasi (`release_year`, `ram`, `original_price`, dll.) diisi otomatis dari katalog jika varian dikenal.
```json
{"brand":"Apple","model":"iPhone 15 Pro","storage":256,
 "usia_pakai_bulan":6,"kesehatan_baterai":97,"kelengkapan":"fullset","garansi":"ada"}
```
Respons 200:
```json
{
  "estimasi_harga_pasar": 19110000,
  "rentang_estimasi": {"bawah": 18339000, "atas": 20948000},
  "mata_uang": "IDR",
  "perkiraan_galat_persen": 4.7,
  "input_dipakai": {"...": "..."},
  "peringatan": [],
  "model_version": "rf-1.0.0+202611061206"
}
```
- **`estimasi_harga_pasar`** adalah nilai yang diteruskan ke Fuzzy Mamdani. Nama field ini dijaga stabil.
- `rentang_estimasi` berasal dari sebaran prediksi antar-pohon (persentil 10-90), bukan selang kepercayaan statistik.
- `perkiraan_galat_persen` = MAPE hasil cross-validation.

Kode error: **422** input tidak valid atau brand tidak dikenal (isi `detail` berupa teks pesan), **503** model belum dilatih.

## Batasan
- Model hanya valid untuk brand yang ada di data latih dan periode harga saat data dikumpulkan.
- Harga dari marketplace adalah harga penawaran, bukan harga transaksi.
- Model tidak menilai kondisi fisik dari foto (itu tugas CNN) dan tidak memakai kondisi fisik sebagai fitur, supaya tidak terhitung dua kali di Fuzzy.
