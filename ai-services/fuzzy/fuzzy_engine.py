"""Mesin Fuzzy Mamdani (numpy murni, tanpa library fuzzy) agar setiap langkah bisa dijelaskan di laporan.

Langkah:
  1. Fuzzifikasi   : nilai tegas -> derajat keanggotaan tiap himpunan input.
  2. Inferensi     : kekuatan aturan = MIN(derajat kondisi, derajat harga)   (operator AND)
                     keluaran aturan = himpunan output yang dipotong (clip) pada kekuatan aturan.
  3. Agregasi      : gabungan semua keluaran aturan dengan MAX.
  4. Defuzzifikasi : titik pusat (centroid) -> rasio harga (%).
"""
import numpy as np

import config as C


def trapmf(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    left = (x - a) / (b - a) if b > a else (x >= a).astype(float)
    right = (d - x) / (d - c) if d > c else (x <= d).astype(float)
    return np.clip(np.minimum(left, right), 0.0, 1.0)


def trimf(x, a, b, c):
    return trapmf(x, a, b, b, c)


def membership(spec, x):
    kind, p = spec
    if kind == "tri":
        return trimf(x, *p)
    if kind == "trap":
        return trapmf(x, *p)
    raise ValueError(f"Jenis fungsi keanggotaan tidak dikenal: {kind}")


class MamdaniEngine:
    def __init__(self):
        lo, hi = C.RASIO_UNIVERSE
        self.grid = np.arange(lo, hi + C.GRID_STEP / 2, C.GRID_STEP)
        self.out_mf = {name: membership(spec, self.grid) for name, spec in C.RASIO_SETS.items()}
        self._check_rules()

    def _check_rules(self):
        for k, h, o in C.RULES:
            assert k in C.KONDISI_SETS and h in C.HARGA_SETS and o in C.RASIO_SETS, f"Aturan tidak valid: {(k, h, o)}"

    # -------------------------------------------------------------- 1. fuzzifikasi
    def fuzzify(self, skor_kondisi: float, harga: float, use_price: bool = True):
        k = float(np.clip(skor_kondisi, *C.KONDISI_UNIVERSE))
        mu_k = {n: float(membership(s, k)) for n, s in C.KONDISI_SETS.items()}
        if use_price:
            h = float(np.clip(harga, *C.HARGA_UNIVERSE))
            mu_h = {n: float(membership(s, h)) for n, s in C.HARGA_SETS.items()}
        else:   # ablasi: abaikan harga (dianggap segmen Menengah penuh)
            mu_h = {n: (1.0 if n == "Menengah" else 0.0) for n in C.HARGA_SETS}
        return mu_k, mu_h

    # -------------------------------------------------------------- 2-4. inferensi sampai defuzzifikasi
    def infer(self, skor_kondisi: float, harga: float, use_price: bool = True) -> dict:
        mu_k, mu_h = self.fuzzify(skor_kondisi, harga, use_price)
        aggregated = np.zeros_like(self.grid)
        fired = []
        for kond, hrg, out in C.RULES:
            strength = min(mu_k[kond], mu_h[hrg])
            if strength <= 0.0:
                continue
            aggregated = np.maximum(aggregated, np.minimum(strength, self.out_mf[out]))
            fired.append({"aturan": f"IF kondisi={kond} AND harga={hrg} THEN rasio={out}",
                          "kekuatan": round(strength, 4)})
        area = aggregated.sum()
        if area <= 0.0:   # seharusnya tidak terjadi bila himpunan input menutup seluruh semesta
            raise ValueError("Tidak ada aturan yang aktif; periksa himpunan fuzzy di config.py")
        rasio = float((self.grid * aggregated).sum() / area)
        fired.sort(key=lambda r: r["kekuatan"], reverse=True)
        return {"rasio": rasio, "mu_kondisi": mu_k, "mu_harga": mu_h,
                "aturan_aktif": fired, "agregasi": aggregated}

    def recommend(self, estimasi_harga: float, skor_kondisi: float, use_price: bool = True) -> dict:
        r = self.infer(skor_kondisi, estimasi_harga, use_price)
        harga_rek = estimasi_harga * r["rasio"] / 100.0
        r["harga_rekomendasi"] = int(round(harga_rek / C.ROUND_TO) * C.ROUND_TO)
        return r
