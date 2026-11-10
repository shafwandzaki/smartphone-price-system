"""API prediksi harga pasar smartphone bekas.

Jalankan:  uvicorn main:app --reload --port 8001
Dokumentasi interaktif: http://127.0.0.1:8001/docs
"""
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config as C
from data_utils import key, normalize_brand, normalize_free_text, normalize_model

app = FastAPI(title="API Prediksi Harga Smartphone Bekas", version=C.MODEL_VERSION)
BUNDLE = None


def load_bundle():
    global BUNDLE
    BUNDLE = joblib.load(C.MODEL_PATH) if C.MODEL_PATH.exists() else None


load_bundle()


class PredictRequest(BaseModel):
    # Wajib: identitas varian
    brand: str
    model: str
    storage: int = Field(gt=0, description="Kapasitas penyimpanan (GB)")
    # Opsional: jika kosong, diisi dari katalog data latih
    release_year: Optional[int] = Field(default=None, ge=2005, le=2035)
    ram: Optional[int] = Field(default=None, gt=0)
    original_price: Optional[float] = Field(default=None, gt=0, description="Harga baru/awal (Rp)")
    battery_capacity: Optional[float] = Field(default=None, gt=0)
    processor_name: Optional[str] = None
    # Opsional: kondisi unit (dipakai jika model dilatih dengan fitur ini)
    usia_pakai_bulan: Optional[int] = Field(default=None, ge=0, le=120)
    kesehatan_baterai: Optional[float] = Field(default=None, ge=0, le=100)
    kelengkapan: Optional[str] = None
    garansi: Optional[str] = None


def _py(v):
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def lookup_defaults(catalog, brand, model_name, storage):
    """Cari spesifikasi bawaan: persis (brand+model+storage), atau varian terdekat dari model yang sama."""
    same = [r for r in catalog if key(r["brand"]) == key(brand) and key(r["model"]) == key(model_name)]
    for r in same:
        if r["storage"] == storage:
            return dict(r), "exact"
    if same:
        near = dict(min(same, key=lambda r: abs(r["storage"] - storage)))
        near["original_price"] = None      # harga awal berbeda per storage: harus diisi
        return near, "model"
    return None, "none"


@app.get("/health")
def health():
    if BUNDLE is None:
        return {"status": "ok", "model_loaded": False}
    m = BUNDLE["meta"]
    return {"status": "ok", "model_loaded": True, "model_version": m["model_version"],
            "trained_at": m["trained_at"], "n_rows": m["data_info"]["rows_clean"],
            "features": BUNDLE["numeric_features"] + BUNDLE["categorical_features"],
            "peringatan_model": m["warnings"]}


@app.get("/catalog")
def catalog():
    """Daftar brand/model/storage yang dikenal model (untuk dropdown di website)."""
    if BUNDLE is None:
        raise HTTPException(503, "Model belum tersedia. Jalankan train.py dulu.")
    out = {}
    for r in BUNDLE["catalog"]:
        out.setdefault(r["brand"], {}).setdefault(r["model"], []).append(r["storage"])
    return {"brands": [
        {"brand": b, "models": [{"model": m, "storages": sorted(s)} for m, s in sorted(ms.items())]}
        for b, ms in sorted(out.items())]}


@app.post("/predict")
def predict(req: PredictRequest):
    if BUNDLE is None:
        raise HTTPException(503, "Model belum tersedia. Jalankan train.py dulu.")
    cat_list = BUNDLE["catalog"]
    num, cat = BUNDLE["numeric_features"], BUNDLE["categorical_features"]
    features = num + cat
    meta = BUNDLE["meta"]

    brands = {key(r["brand"]): r["brand"] for r in cat_list}
    brand = normalize_brand(req.brand)
    if key(brand) not in brands:
        raise HTTPException(422, f"Brand '{req.brand}' belum didukung. Brand yang dikenal: {sorted(brands.values())}")
    brand = brands[key(brand)]
    model_name = normalize_model(req.model)

    defaults, level = lookup_defaults(cat_list, brand, model_name, req.storage)
    if defaults and level == "exact":
        model_name = defaults["model"]
    warnings = []
    if level == "none":
        warnings.append(f"Model '{model_name}' belum ada di data latih; estimasi kurang andal.")
    elif level == "model":
        warnings.append(f"Storage {req.storage} GB untuk '{model_name}' belum ada di data latih; "
                        "spesifikasi diambil dari varian terdekat.")

    given = req.model_dump()
    given.update(brand=brand, model=model_name)
    for f in ("processor_name",):
        given[f] = normalize_free_text(given.get(f))
    for f in ("kelengkapan", "garansi"):
        given[f] = normalize_free_text(given.get(f), lower=True)

    row = {}
    for f in features:
        v = given.get(f)
        if (v is None or (not isinstance(v, str) and pd.isna(v))) and defaults:
            v = defaults.get(f)
        row[f] = v

    missing = [f for f in C.REQUIRED_COLUMNS if f in features and _py(row.get(f)) is None]
    if missing:
        raise HTTPException(422, f"Varian ini tidak ada di katalog; lengkapi field berikut: {missing}")
    unfilled = [f for f in C.USAGE_FEATURES if f in features and _py(row.get(f)) is None]
    if unfilled:
        warnings.append(f"Field kondisi {unfilled} tidak diisi; memakai nilai tipikal dari data latih.")

    X = pd.DataFrame([row])[features]
    for c in cat:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    price = float(BUNDLE["model"].predict(X)[0])

    # Rentang dari sebaran prediksi antar-pohon (bukan selang kepercayaan statistik).
    fitted = BUNDLE["model"].regressor_
    Xt = fitted.named_steps["prep"].transform(X)
    tree_preds = np.expm1([t.predict(Xt)[0] for t in fitted.named_steps["reg"].estimators_])
    lo, hi = np.percentile(tree_preds, [10, 90])

    mape = (meta.get("metrics_cv_randomforest") or {}).get("MAPE_%")
    return {
        "estimasi_harga_pasar": int(round(price, -3)),
        "rentang_estimasi": {"bawah": int(round(lo, -3)), "atas": int(round(hi, -3))},
        "mata_uang": "IDR",
        "perkiraan_galat_persen": round(mape, 1) if mape is not None else None,
        "input_dipakai": {k: _py(v) for k, v in row.items()},
        "peringatan": warnings,
        "model_version": meta["model_version"],
    }
