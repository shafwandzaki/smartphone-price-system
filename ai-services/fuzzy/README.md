# Fuzzy Mamdani – Rekomendasi Harga Jual Smartphone Bekas

Layanan AI untuk Penelitian 3. API FastAPI di port **8003**. Menggabungkan:
- **estimasi harga pasar** (Rp) dari Random Forest (port 8001), dan
- **skor kondisi fisik** (0-100, **100 = terbaik**) dari CNN (port 8002),

menjadi **harga rekomendasi jual**. Fuzzy Mamdani berbasis aturan: tidak dilatih seperti RF atau CNN,
sehingga validasinya harus dengan data acuan (lihat "Validasi").

## Struktur
```
fuzzy-mamdani/
├── config.py            # himpunan fuzzy, fungsi keanggotaan, aturan (ubah parameter di sini saja)
├── fuzzy_engine.py      # mesin Mamdani (numpy murni, tiap langkah bisa dijelaskan)
├── main.py              # API: /health, /rules, /recommend
├── evaluate.py          # validasi terhadap harga acuan + ablasi + pembanding
├── plot_membership.py   # gambar untuk laporan
├── tests/               # smoke_test.py dan reference_template.csv
└── reports/             # gambar dan hasil validasi
```

## Setup (Windows)
```cmd
cd ai-services\fuzzy-mamdani
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
python tests\smoke_test.py            # harus berakhir "SEMUA LOLOS"
python plot_membership.py             # gambar untuk laporan -> reports\
uvicorn main:app --reload --port 8003
```
Uji interaktif di `http://127.0.0.1:8003/docs` (POST /recommend, Try it out).

## Cara kerja (untuk bab metodologi)
1. **Fuzzifikasi**: skor kondisi dan estimasi harga diubah menjadi derajat keanggotaan.
2. **Inferensi**: kekuatan aturan = MIN(derajat kondisi, derajat harga). Keluaran aturan = himpunan output dipotong pada kekuatan itu.
3. **Agregasi**: semua keluaran aturan digabung dengan MAX.
4. **Defuzzifikasi**: titik pusat (centroid) menghasilkan **rasio** (% dari estimasi pasar).

`harga_rekomendasi = estimasi_harga_pasar x rasio / 100` (dibulatkan ke Rp10.000).

Keluaran berupa rasio, bukan harga mutlak, supaya aturan berlaku untuk semua rentang harga.
Dalam bentuk mutlak, contoh aturan di dokumen proyek tetap terpenuhi: ponsel mahal yang rusak
(rasio Rendah) harganya tetap lebih tinggi daripada ponsel murah yang kondisinya baik.

### Himpunan fuzzy (nilai awal, lihat config.py)
| Variabel | Himpunan |
|---|---|
| Skor kondisi (0-100) | Buruk, Sedang, Baik, SangatBaik |
| Estimasi harga (Rp) | Murah (sampai 6 juta), Menengah (3-15 juta), Mahal (di atas 10 juta) |
| Rasio harga (%) | SangatRendah (~44), Rendah (~58), Sedang (~72), Tinggi (~86), SangatTinggi (~95) |

### Aturan (12 aturan)
| Kondisi \ Harga | Murah | Menengah | Mahal |
|---|---|---|---|
| SangatBaik | SangatTinggi | SangatTinggi | SangatTinggi |
| Baik | Tinggi | Tinggi | Tinggi |
| Sedang | Rendah | Sedang | Sedang |
| Buruk | SangatRendah | Rendah | Rendah |

Hipotesis di balik aturan: saat kondisi baik, harga tidak memengaruhi rasio; saat kondisi buruk,
perangkat yang lebih mahal mempertahankan nilai relatif lebih tinggi. **Ini hipotesis awal, bukan fakta pasar**;
buktikan atau ubah lewat validasi.

## Kontrak API

### `POST /recommend`
```json
{
  "estimasi_harga_pasar": 17500000,
  "skor_kondisi": 45.33,
  "rentang_estimasi": {"bawah": 16000000, "atas": 19000000}
}
```
`rentang_estimasi` opsional (diambil dari respons Random Forest).

Respons 200 (keluaran nyata dari parameter awal untuk input di atas):
```json
{
  "harga_rekomendasi": 10880000,
  "faktor_harga_persen": 62.2,
  "rentang_rekomendasi": {
    "bawah": 9950000,
    "atas": 11820000
  },
  "mata_uang": "IDR",
  "kondisi_dominan": "Buruk",
  "derajat_keanggotaan": {
    "kondisi": {
      "Buruk": 0.645,
      "Sedang": 0.242,
      "Baik": 0.0,
      "SangatBaik": 0.0
    },
    "harga": {
      "Murah": 0.0,
      "Menengah": 0.0,
      "Mahal": 0.938
    }
  },
  "aturan_aktif": [
    {
      "aturan": "IF kondisi=Buruk AND harga=Mahal THEN rasio=Rendah",
      "kekuatan": 0.6447
    },
    {
      "aturan": "IF kondisi=Sedang AND harga=Mahal THEN rasio=Sedang",
      "kekuatan": 0.2423
    }
  ],
  "input": {
    "estimasi_harga_pasar": 17500000.0,
    "skor_kondisi": 45.33
  },
  "peringatan": [],
  "model_version": "fuzzy-1.0.0"
}
```
`aturan_aktif` dan `derajat_keanggotaan` membuat hasil bisa dijelaskan kepada pengguna dan penguji.

Kode error: **422** input tidak valid (skor di luar 0-100, harga ≤ 0, rentang terbalik).
Endpoint lain: `GET /health`, `GET /rules` (seluruh himpunan dan aturan).

## Alur integrasi (Laravel sebagai orkestrator)
```
Form pengguna ─┬─> RF  POST :8001/predict        -> estimasi_harga_pasar, rentang_estimasi
               ├─> CNN POST :8002/predict-image   -> skor_kondisi (0-100, 100 = terbaik)
               └─> Fuzzy POST :8003/recommend     <- gabungan keduanya -> harga_rekomendasi
```
Jika CNN mengeluarkan skor kerusakan (tinggi = rusak), ubah dulu: `skor_kondisi = 100 - skor_kerusakan`.

## Validasi (wajib untuk laporan)
1. Kumpulkan **data acuan**, minimal 30 baris, idealnya lebih: harga nyata ponsel bekas **dengan kerusakan serupa**
   (listing marketplace) atau penilaian beberapa pakar/penjual. Catat sumbernya di kolom `keterangan`.
2. Untuk tiap unit isi `estimasi_harga_pasar` (dari RF) dan `skor_kondisi` (dari CNN), lalu `harga_acuan`.
   Gunakan `tests/reference_template.csv`, simpan sebagai `tests/reference.csv`.
3. `python evaluate.py --ref tests\reference.csv` menampilkan MAE, RMSE, MAPE, bias, dan persentase dalam ±10%
   untuk: Fuzzy lengkap, Fuzzy tanpa input harga (ablasi), pembanding `harga x skor/100`, dan tanpa penyesuaian.
4. Kalibrasi parameter di `config.py` (himpunan dan aturan) berdasarkan hasil, lalu ulangi.
   Hindari menyetel dan melaporkan pada data yang sama; sisihkan sebagian acuan untuk pelaporan akhir.

Jika Fuzzy lengkap tidak lebih baik daripada versi "kondisi saja", input harga belum memberi nilai tambah,
dan itu temuan yang sah untuk dibahas di laporan.

## Asumsi dan batasan
- **Semua parameter masih usulan awal** (batas himpunan, rentang rasio, aturan) dan belum dikalibrasi dengan data pasar.
- **Kondisi tidak dihitung dua kali**: diasumsikan RF dilatih dari listing berkondisi fisik normal/baik atau tidak memuat
  efek kerusakan layar. Jika dataset RF berisi listing ponsel retak, rasio fuzzy akan menurunkan harga dua kali.
- Kualitas keluaran dibatasi kualitas masukan: skor CNN belum divalidasi, dan RF memakai harga penawaran, bukan transaksi.
- Rasio terendah yang dapat dihasilkan sekitar 44% (titik pusat himpunan SangatRendah) dan tertinggi sekitar 95%.
- Fuzzy tidak menilai kerusakan internal, baterai, atau fungsi komponen.
