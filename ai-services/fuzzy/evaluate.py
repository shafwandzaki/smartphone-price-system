"""Validasi rekomendasi Fuzzy terhadap HARGA ACUAN yang dapat dipertanggungjawabkan.

File acuan (CSV) wajib berkolom:
    estimasi_harga_pasar   estimasi dari Random Forest untuk unit tsb (Rp)
    skor_kondisi           skor dari CNN, 0-100 (100 = terbaik)
    harga_acuan            harga pembanding nyata untuk unit berkondisi serupa (Rp)
kolom opsional: keterangan (sumber/tautan/catatan)

Contoh sumber harga_acuan: listing nyata ponsel bekas dengan kerusakan serupa, atau penilaian
beberapa pakar/penjual. Isi tests/reference_template.csv lalu simpan sebagai tests/reference.csv.

Pakai:  python evaluate.py --ref tests/reference.csv
"""
import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from fuzzy_engine import MamdaniEngine

REQUIRED = ["estimasi_harga_pasar", "skor_kondisi", "harga_acuan"]


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    return {"MAE_Rp": float(np.abs(err).mean()), "RMSE_Rp": float(np.sqrt((err ** 2).mean())),
            "MAPE_%": float(np.mean(np.abs(err / y)) * 100),
            "Bias_%": float(np.mean(err / y) * 100),          # + berarti rekomendasi cenderung terlalu tinggi
            "dalam_10%": float(np.mean(np.abs(err / y) <= 0.10) * 100)}


def main(a):
    df = pd.read_csv(a.ref, encoding="utf-8-sig")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: kolom wajib tidak ada: {missing}")
    df = df.dropna(subset=REQUIRED)
    df = df[(df.estimasi_harga_pasar > 0) & (df.harga_acuan > 0) & df.skor_kondisi.between(0, 100)]
    if len(df) < 5:
        sys.exit("ERROR: data acuan terlalu sedikit (minimal 5 baris, idealnya 30 ke atas).")
    eng = MamdaniEngine()
    p_full = [eng.recommend(h, s)["harga_rekomendasi"] for h, s in zip(df.estimasi_harga_pasar, df.skor_kondisi)]
    p_cond = [eng.recommend(h, s, use_price=False)["harga_rekomendasi"] for h, s in zip(df.estimasi_harga_pasar, df.skor_kondisi)]
    preds = {
        "Fuzzy Mamdani (kondisi + harga)": p_full,
        "Fuzzy Mamdani (kondisi saja)": p_cond,
        "Pembanding: harga x skor/100": (df.estimasi_harga_pasar * df.skor_kondisi / 100).tolist(),
        "Pembanding: tanpa penyesuaian": df.estimasi_harga_pasar.tolist(),
    }
    res = pd.DataFrame({k: metrics(df.harga_acuan, v) for k, v in preds.items()}).T
    print(f"\n=== Validasi terhadap harga acuan ({len(df)} baris) ===")
    print(res.round(1).to_string())
    if len(df) < 30:
        print("\nPERINGATAN: kurang dari 30 baris acuan; hasil belum stabil.")
    print("\nBias_% positif = rekomendasi cenderung terlalu tinggi; negatif = terlalu rendah.")
    print("Jika kedua versi Fuzzy hampir sama, input harga belum memberi nilai tambah (cek aturan harga).")

    C.REPORT_DIR.mkdir(exist_ok=True)
    out = df.copy()
    out["rekomendasi_fuzzy"] = p_full
    out["selisih_%"] = (out["rekomendasi_fuzzy"] - out["harga_acuan"]) / out["harga_acuan"] * 100
    out.to_csv(C.REPORT_DIR / "validation_detail.csv", index=False)
    res.to_csv(C.REPORT_DIR / "validation_summary.csv")
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(df.harga_acuan / 1e6, np.array(p_full) / 1e6, alpha=0.7)
    lim = [0, max(df.harga_acuan.max(), max(p_full)) / 1e6 * 1.05]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("Harga acuan (juta Rp)"); ax.set_ylabel("Rekomendasi Fuzzy (juta Rp)")
    ax.set_title("Rekomendasi vs harga acuan"); fig.tight_layout()
    fig.savefig(C.REPORT_DIR / "validation_scatter.png", dpi=150); plt.close(fig)
    print(f"\nHasil disimpan di {C.REPORT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="tests/reference.csv")
    main(ap.parse_args())
