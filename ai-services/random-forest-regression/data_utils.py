"""Pemuatan dan pembersihan data (dipakai train.py dan main.py)."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config as C


# ------------------------------------------------------------------ parsing
def parse_number(x):
    """'43,999,000' / 'Rp 5.500.000' / '98%' / 2048 -> float. Gagal -> NaN."""
    if x is None or (not isinstance(x, str) and pd.isna(x)):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = re.sub(r"[^0-9.,\-]", "", str(x))
    if s in ("", "-"):
        return np.nan
    if re.fullmatch(r"-?\d{1,3}([.,]\d{3})+", s):   # pemisah ribuan
        s = re.sub(r"[.,]", "", s)
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def normalize_brand(v):
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return np.nan
    return re.sub(r"\s+", " ", str(v)).strip().title()


def normalize_model(v):
    """'Iphone 17 pro max' -> 'iPhone 17 Pro Max' (tidak mengubah huruf kapital yang sudah ada)."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return np.nan
    words = re.sub(r"\s+", " ", str(v)).strip().split(" ")
    out = []
    for w in words:
        out.append("iPhone" if w.lower() == "iphone" else w[:1].upper() + w[1:])
    return " ".join(out)


def normalize_free_text(v, lower=False):
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return np.nan
    s = re.sub(r"\s+", " ", str(v)).strip()
    return s.lower() if lower else s


def key(s):
    """Kunci pencocokan tanpa peduli huruf besar/kecil."""
    return str(s).casefold().strip()


# ------------------------------------------------------------------ memuat
def load_dataset(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File data tidak ditemukan: {path}")
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [re.sub(r"\s+", "_", str(c).strip().lower()) for c in df.columns]
    return df.rename(columns=C.COLUMN_ALIASES)


def clean_dataframe(df: pd.DataFrame):
    """Mengembalikan (df_bersih, info). Error jelas jika kolom wajib tidak ada."""
    df = df.copy()
    missing_cols = [c for c in C.REQUIRED_COLUMNS + [C.TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Kolom wajib tidak ada di dataset: {missing_cols}. "
            f"Kolom yang terbaca: {list(df.columns)}"
        )
    df = df.drop(columns=[c for c in C.DROP_COLUMNS if c in df.columns])

    for c in set(C.NUMERIC_CANDIDATES + [C.TARGET]):
        if c in df.columns:
            df[c] = df[c].map(parse_number)

    df["brand"] = df["brand"].map(normalize_brand)
    df["model"] = df["model"].map(normalize_model)
    if "processor_name" in df.columns:
        df["processor_name"] = df["processor_name"].map(normalize_free_text)
    for c in ("kelengkapan", "garansi"):
        if c in df.columns:
            df[c] = df[c].map(lambda v: normalize_free_text(v, lower=True))

    n0 = len(df)
    df = df.dropna(subset=C.REQUIRED_COLUMNS + [C.TARGET])
    df = df[df[C.TARGET] > 0]
    n_invalid = n0 - len(df)
    n1 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    info = {"rows_raw": n0, "rows_dropped_invalid": n_invalid,
            "rows_dropped_duplicate": n1 - len(df), "rows_clean": len(df)}
    return df, info


def active_features(df: pd.DataFrame):
    """Fitur yang dipakai = kandidat yang ada di data, terisi, dan bervariasi."""
    def usable(c):
        return c in df.columns and df[c].notna().any() and df[c].nunique(dropna=True) > 1
    num = [c for c in C.NUMERIC_CANDIDATES if usable(c)]
    cat = [c for c in C.CATEGORICAL_CANDIDATES if usable(c)]
    return num, cat


def build_catalog(df: pd.DataFrame):
    """Katalog (brand, model, storage) -> spesifikasi bawaan, untuk dropdown web & pengisian otomatis."""
    records = []
    for (b, m, s), g in df.groupby(["brand", "model", "storage"]):
        rec = {"brand": b, "model": m, "storage": int(s), "n_rows": int(len(g))}
        for f in C.SPEC_NUMERIC:
            if f in g.columns:
                v = g[f].median()
                rec[f] = None if pd.isna(v) else float(v)
        for f in C.SPEC_CATEGORICAL:
            if f in g.columns:
                mode = g[f].dropna().mode()
                rec[f] = str(mode.iloc[0]) if len(mode) else None
        records.append(rec)
    return records
