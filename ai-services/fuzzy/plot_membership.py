"""Membuat gambar untuk laporan: fungsi keanggotaan dan permukaan kendali.   python plot_membership.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C
from fuzzy_engine import MamdaniEngine, membership


def panel(ax, universe, sets, title, xlabel, scale=1.0):
    x = np.linspace(universe[0], universe[1], 1000)
    for name, spec in sets.items():
        ax.plot(x / scale, membership(spec, x), label=name, lw=2)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("Derajat keanggotaan")
    ax.set_ylim(-0.02, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=8)


def main():
    C.REPORT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    panel(axes[0], C.KONDISI_UNIVERSE, C.KONDISI_SETS, "Input: skor kondisi fisik", "Skor kondisi (100 = terbaik)")
    panel(axes[1], C.HARGA_UNIVERSE, C.HARGA_SETS, "Input: estimasi harga pasar", "Harga (juta Rp)", 1e6)
    panel(axes[2], C.RASIO_UNIVERSE, C.RASIO_SETS, "Output: rasio harga rekomendasi", "Rasio terhadap harga pasar (%)")
    fig.tight_layout(); fig.savefig(C.REPORT_DIR / "membership_functions.png", dpi=150); plt.close(fig)

    eng = MamdaniEngine()
    scores = np.linspace(0, 100, 101)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for h, lab in [(3_000_000, "Rp3 juta"), (8_000_000, "Rp8 juta"), (20_000_000, "Rp20 juta")]:
        ax.plot(scores, [eng.infer(s, h)["rasio"] for s in scores], lw=2, label=f"harga pasar {lab}")
    ax.set_xlabel("Skor kondisi (100 = terbaik)"); ax.set_ylabel("Rasio harga rekomendasi (%)")
    ax.set_title("Permukaan kendali: skor kondisi -> rasio"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.REPORT_DIR / "control_curve.png", dpi=150); plt.close(fig)
    print(f"Gambar disimpan di {C.REPORT_DIR}")


if __name__ == "__main__":
    main()
