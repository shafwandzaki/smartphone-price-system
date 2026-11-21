"""Uji otomatis sifat dasar sistem fuzzy.   python tests/smoke_test.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from fastapi.testclient import TestClient

import config as C
import main
from fuzzy_engine import MamdaniEngine

eng = MamdaniEngine()
ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")


# 1. Monoton: kondisi lebih baik tidak boleh menurunkan rasio (untuk berbagai harga)
worst = 0.0
for h in [1e6, 3e6, 5e6, 8e6, 12e6, 20e6, 35e6]:
    r = np.array([eng.infer(s, h)["rasio"] for s in np.linspace(0, 100, 201)])
    worst = min(worst, float(np.diff(r).min()))
check("Rasio naik monoton terhadap skor kondisi", worst > -1e-6, f"(penurunan terbesar {worst:.4f})")

# 2. Kontinu: tidak ada lompatan tajam antar skor berdekatan
jump = max(float(np.abs(np.diff([eng.infer(s, 10e6)["rasio"] for s in np.linspace(0, 100, 1001)])).max()), 0)
check("Keluaran kontinu (tanpa lompatan)", jump < 0.5, f"(lompatan terbesar per 0,1 skor = {jump:.3f} poin rasio)")

# 3. Rentang keluaran
rs = [eng.infer(s, h)["rasio"] for s in (0, 100) for h in (1e6, 40e6)]
check("Rasio di dalam rentang semesta output", min(rs) >= C.RASIO_UNIVERSE[0] and max(rs) <= C.RASIO_UNIVERSE[1],
      f"({min(rs):.1f}% sampai {max(rs):.1f}%)")

# 4. Seluruh semesta input tercakup (selalu ada aturan aktif)
good = all(eng.infer(s, h)["aturan_aktif"] for s in np.linspace(0, 100, 41) for h in np.linspace(0, 45e6, 46))
check("Selalu ada aturan aktif untuk semua kombinasi input", good)

# 5. Perangkat mahal rusak mempertahankan rasio >= perangkat murah rusak (sesuai hipotesis aturan)
check("Hipotesis harga: kondisi buruk, mahal >= murah",
      eng.infer(20, 25e6)["rasio"] >= eng.infer(20, 2e6)["rasio"])

# 6. API
c = TestClient(main.app)
r = c.post("/recommend", json={"estimasi_harga_pasar": 17_500_000, "skor_kondisi": 45.33,
                               "rentang_estimasi": {"bawah": 16_000_000, "atas": 19_000_000}})
j = r.json()
check("API /recommend 200 dan harga masuk akal", r.status_code == 200 and 0 < j["harga_rekomendasi"] < 17_500_000,
      f"(Rp{j['harga_rekomendasi']:,} = {j['faktor_harga_persen']}%)")
check("Rentang ikut disesuaikan", j["rentang_rekomendasi"]["bawah"] < j["rentang_rekomendasi"]["atas"])
check("Validasi: skor di luar 0-100 ditolak", c.post("/recommend", json={"estimasi_harga_pasar": 1e6, "skor_kondisi": 120}).status_code == 422)
check("Validasi: harga <= 0 ditolak", c.post("/recommend", json={"estimasi_harga_pasar": 0, "skor_kondisi": 50}).status_code == 422)
check("Validasi: rentang terbalik ditolak", c.post("/recommend", json={"estimasi_harga_pasar": 1e6, "skor_kondisi": 50,
      "rentang_estimasi": {"bawah": 2e6, "atas": 1e6}}).status_code == 422)
check("/health dan /rules", c.get("/health").status_code == 200 and len(c.get("/rules").json()["aturan"]) == len(C.RULES))
print("\nSEMUA LOLOS" if ok else "\nADA YANG GAGAL")
sys.exit(0 if ok else 1)
