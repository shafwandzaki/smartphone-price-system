"""Training Random Forest Regression harga smartphone bekas.

Pakai:
    python train.py                      # membaca dataset/dataset_harga.csv
    python train.py --data path/file.xlsx
    python train.py --n-iter 40          # tuning lebih banyak
    python train.py --no-plots
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold,
                                     RandomizedSearchCV, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import config as C
from data_utils import active_features, build_catalog, clean_dataframe, load_dataset


# ------------------------------------------------------------------ model
def make_model(num, cat, regressor):
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])
    pipe = Pipeline([("prep", prep), ("reg", regressor)])
    # Harga cenderung miring ke kanan -> dilatih pada skala log.
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_%": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def aggregated_importance(fitted, num, cat):
    pipe = fitted.regressor_
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = pipe.named_steps["reg"].feature_importances_
    agg = {f: 0.0 for f in num + cat}
    for n, v in zip(names, imp):
        if n.startswith("num__"):
            agg[n[5:]] += float(v)
        else:
            rest = n[5:]
            for f in cat:
                if rest.startswith(f + "_"):
                    agg[f] += float(v)
                    break
    return dict(sorted(agg.items(), key=lambda kv: kv[1], reverse=True))


# ------------------------------------------------------------------ plot
def save_plots(y_te, pred_te, importance):
    C.REPORT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(np.asarray(y_te) / 1e6, np.asarray(pred_te) / 1e6, alpha=0.7)
    lim = [min(y_te.min(), pred_te.min()) / 1e6, max(y_te.max(), pred_te.max()) / 1e6]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("Harga sebenarnya (juta Rp)")
    ax.set_ylabel("Harga prediksi (juta Rp)")
    ax.set_title("Prediksi vs sebenarnya (data uji)")
    fig.tight_layout()
    fig.savefig(C.REPORT_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    items = list(importance.items())[::-1]
    ax.barh([k for k, _ in items], [v for _, v in items])
    ax.set_xlabel("Tingkat kepentingan")
    ax.set_title("Feature importance")
    fig.tight_layout()
    fig.savefig(C.REPORT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ utama
def main(args):
    data_path = args.data or C.DATA_PATH
    try:
        df, info = clean_dataframe(load_dataset(data_path))
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"ERROR: {e}")

    n = len(df)
    print(f"Data: {data_path.name if hasattr(data_path, 'name') else data_path}")
    print(f"  baris mentah {info['rows_raw']} | dibuang tidak valid {info['rows_dropped_invalid']} "
          f"| dibuang duplikat {info['rows_dropped_duplicate']} | dipakai {n}")
    if n < 10:
        sys.exit("ERROR: data terlalu sedikit untuk dilatih (minimal 10 baris).")

    num, cat = active_features(df)
    features = num + cat
    print(f"  fitur dipakai: {features}")

    warnings = []
    if not any(f in features for f in C.USAGE_FEATURES):
        warnings.append("Dataset belum memuat fitur kondisi/pemakaian (usia pakai, kesehatan baterai, "
                        "kelengkapan, garansi). Model hanya membedakan varian, bukan kondisi unit.")
    per_variant = n / df.groupby(["brand", "model", "storage"]).ngroups
    if per_variant < 2:
        warnings.append("Rata-rata satu varian (brand-model-storage) hanya punya satu harga. "
                        "Skor evaluasi belum mencerminkan pemakaian nyata dan tidak layak untuk laporan.")
    if n < 100:
        warnings.append(f"Jumlah data {n} baris tergolong kecil; hasil evaluasi belum stabil.")
    for w in warnings:
        print(f"  PERINGATAN: {w}")

    X, y = df[features].copy(), df[C.TARGET]
    for c in cat:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)

    groups = None
    if C.GROUP_COL in df.columns and df[C.GROUP_COL].notna().any():
        groups = df[C.GROUP_COL].astype(object)
        groups = groups.where(groups.notna(), pd.Series([f"_row{i}" for i in range(n)]))
        print(f"  split berbasis grup ({C.GROUP_COL}) untuk mencegah leakage")
    else:
        print(f"  kolom '{C.GROUP_COL}' tidak ada: split acak biasa")

    idx = np.arange(n)
    if groups is not None:
        tr, te = next(GroupShuffleSplit(1, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE)
                      .split(X, y, groups))
    else:
        tr, te = train_test_split(idx, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE)
    X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    g_tr = groups.iloc[tr] if groups is not None else None

    # --- pembanding
    results = {}
    for name, reg in {"Baseline_rata2": DummyRegressor(strategy="mean"),
                      "LinearRegression": LinearRegression()}.items():
        results[name] = metrics(y_te, make_model(num, cat, reg).fit(X_tr, y_tr).predict(X_te))

    # --- Random Forest + tuning
    rf = make_model(num, cat, RandomForestRegressor(random_state=C.RANDOM_STATE))
    param_dist = {
        "regressor__reg__n_estimators": [100, 200, 400],
        "regressor__reg__max_depth": [None, 8, 12, 20],
        "regressor__reg__min_samples_leaf": [1, 2, 4],
        "regressor__reg__max_features": ["sqrt", 0.5, 1.0],
    }
    if groups is not None:
        cv_tune = GroupKFold(n_splits=min(5, g_tr.nunique()))
    else:
        cv_tune = KFold(n_splits=min(5, len(X_tr)), shuffle=True, random_state=C.RANDOM_STATE)
    search = RandomizedSearchCV(rf, param_dist, n_iter=args.n_iter, cv=cv_tune,
                                scoring="neg_mean_absolute_error",
                                random_state=C.RANDOM_STATE, n_jobs=-1)
    search.fit(X_tr, y_tr, groups=g_tr) if groups is not None else search.fit(X_tr, y_tr)
    best = search.best_estimator_
    pred_te = best.predict(X_te)
    results["RandomForest"] = metrics(y_te, pred_te)

    # --- cross-validation seluruh data (lebih stabil untuk data kecil)
    if groups is not None:
        cv_all = GroupKFold(n_splits=min(5, groups.nunique()))
    else:
        cv_all = KFold(n_splits=min(5, n), shuffle=True, random_state=C.RANDOM_STATE)
    oof = cross_val_predict(clone(best), X, y, cv=cv_all, groups=groups, n_jobs=-1)
    cv_metrics = metrics(y, oof)

    print("\n=== Evaluasi data uji (holdout) ===")
    print(pd.DataFrame(results).T.round(2).to_string())
    print("\n=== Random Forest, cross-validation seluruh data ===")
    print({k: round(v, 2) for k, v in cv_metrics.items()})
    print("\nParameter terbaik:", search.best_params_)

    importance = aggregated_importance(best, num, cat)
    print("Feature importance:", {k: round(v, 3) for k, v in importance.items()})
    if not args.no_plots:
        save_plots(y_te, pd.Series(pred_te, index=y_te.index), importance)
        print(f"Grafik disimpan di {C.REPORT_DIR}")

    # --- simpan
    trained_at = datetime.now()
    version = f"{C.MODEL_VERSION}+{trained_at:%Y%m%d%H%M}"
    data_hash = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:12]
    meta = {
        "model_version": version,
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "data_file": str(getattr(data_path, "name", data_path)),
        "data_sha256_12": data_hash,
        "data_info": info,
        "n_brands": int(df["brand"].nunique()), "n_models": int(df["model"].nunique()),
        "numeric_features": num, "categorical_features": cat,
        "metrics_holdout": results, "metrics_cv_randomforest": cv_metrics,
        "feature_importance": importance,
        "best_params": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                        for k, v in search.best_params_.items()},
        "warnings": warnings,
    }
    bundle = {"model": best, "numeric_features": num, "categorical_features": cat,
              "catalog": build_catalog(df), "meta": meta}
    C.MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, C.MODEL_PATH)
    C.META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nModel disimpan : {C.MODEL_PATH.name} (versi {version})")
    print(f"Metadata       : {C.META_PATH.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=lambda p: __import__("pathlib").Path(p), default=None)
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--no-plots", action="store_true")
    main(ap.parse_args())
