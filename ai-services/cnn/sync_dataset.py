import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS = ["train", "valid", "test"]
VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp")

for split in SPLITS:
    split_dir = os.path.join(BASE_DIR, "dataset", split)
    csv_path = os.path.join(split_dir, "_classes.csv")

    if not os.path.exists(csv_path):
        print(f"[{split}] File _classes.csv tidak ditemukan, dilewati.")
        continue

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["filename"] = df["filename"].astype(str).str.strip()

    # 1. Cek semua file gambar fisik yang benar-benar ada di folder
    actual_files = set(
        f for f in os.listdir(split_dir)
        if f.lower().endswith(VALID_EXTS)
    )

    # 2. Hapus baris di CSV yang file gambarnya tidak ada di folder
    before_count = len(df)
    df = df[df["filename"].isin(actual_files)].copy()
    removed_count = before_count - len(df)

    # 3. Cari gambar fisik baru yang belum terdaftar di CSV
    registered_files = set(df["filename"])
    unregistered_files = sorted(list(actual_files - registered_files))

    new_rows = []
    for fname in unregistered_files:
        new_rows.append({
            "filename": fname,
            "Multi fissure": 0,
            "fissure": 0,
            "impact": 0
        })

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    # Simpan kembali CSV yang sudah 100% sinkron
    df.to_csv(csv_path, index=False)
    print(
        f"[{split.upper()}] Sinkron! Dihapus baris hantu: {removed_count} | "
        f"Ditambahkan gambar baru (mulus): {len(new_rows)} | Total data: {len(df)}"
    )