"""API rekomendasi harga jual (Fuzzy Mamdani).

Jalankan:  uvicorn main:app --reload --port 8003
Dokumentasi interaktif: http://127.0.0.1:8003/docs

Alur (dikendalikan Laravel):
  1) POST :8001/predict        -> estimasi_harga_pasar (Random Forest)
  2) POST :8002/predict-image  -> skor_kondisi 0-100, 100 = terbaik (CNN)
  3) POST :8003/recommend      -> harga_rekomendasi (layanan ini)
"""
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config as C
from fuzzy_engine import MamdaniEngine

app = FastAPI(title="API Rekomendasi Harga Jual (Fuzzy Mamdani)", version=C.VERSION)
ENGINE = MamdaniEngine()


class PriceRange(BaseModel):
    bawah: float = Field(gt=0)
    atas: float = Field(gt=0)


class RecommendRequest(BaseModel):
    estimasi_harga_pasar: float = Field(gt=0, description="Rupiah, dari Random Forest")
    skor_kondisi: float = Field(ge=0, le=100, description="0-100, 100 = kondisi terbaik, dari CNN")
    rentang_estimasi: Optional[PriceRange] = Field(
        default=None, description="Opsional, dari Random Forest; ikut disesuaikan dengan rasio yang sama")


def _round_price(v: float) -> int:
    return int(round(v / C.ROUND_TO) * C.ROUND_TO)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": C.VERSION, "n_rules": len(C.RULES),
            "input": {"estimasi_harga_pasar": "Rp", "skor_kondisi": "0-100 (100 = terbaik)"},
            "peringatan_model": [
                "Parameter himpunan fuzzy dan aturan adalah usulan awal yang belum dikalibrasi "
                "dengan data acuan pasar."]}


@app.get("/rules")
def rules():
    return {"himpunan_kondisi": C.KONDISI_SETS, "himpunan_harga": C.HARGA_SETS,
            "himpunan_rasio": C.RASIO_SETS,
            "aturan": [{"kondisi": k, "harga": h, "rasio": o} for k, h, o in C.RULES]}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    if req.rentang_estimasi and req.rentang_estimasi.bawah > req.rentang_estimasi.atas:
        raise HTTPException(422, "rentang_estimasi.bawah tidak boleh lebih besar dari atas.")
    r = ENGINE.recommend(req.estimasi_harga_pasar, req.skor_kondisi)
    warnings = []
    if req.estimasi_harga_pasar > C.HARGA_UNIVERSE[1]:
        warnings.append("Estimasi harga di atas rentang himpunan fuzzy; diperlakukan sebagai segmen Mahal.")
    rentang = None
    if req.rentang_estimasi:
        f = r["rasio"] / 100.0
        rentang = {"bawah": _round_price(req.rentang_estimasi.bawah * f),
                   "atas": _round_price(req.rentang_estimasi.atas * f)}
    dominan = max(r["mu_kondisi"], key=r["mu_kondisi"].get)
    return {
        "harga_rekomendasi": r["harga_rekomendasi"],
        "faktor_harga_persen": round(r["rasio"], 1),
        "rentang_rekomendasi": rentang,
        "mata_uang": "IDR",
        "kondisi_dominan": dominan,
        "derajat_keanggotaan": {
            "kondisi": {k: round(v, 3) for k, v in r["mu_kondisi"].items()},
            "harga": {k: round(v, 3) for k, v in r["mu_harga"].items()}},
        "aturan_aktif": r["aturan_aktif"],
        "input": {"estimasi_harga_pasar": req.estimasi_harga_pasar, "skor_kondisi": req.skor_kondisi},
        "peringatan": warnings,
        "model_version": C.VERSION,
    }
