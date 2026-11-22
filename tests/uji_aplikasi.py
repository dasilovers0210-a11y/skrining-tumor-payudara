"""Uji otomatis 5 skenario (lihat docs/log-pengujian.md).

Jalankan dari folder utama repo:  python tests/uji_aplikasi.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app_flask import app, FEATURES  # noqa: E402

sampel = pd.read_csv(ROOT / "data_uji" / "sampel_uji.csv")
client = app.test_client()


def kirim(baris=None, ubah=None, kosongkan=None):
    data = {"nama": "Pasien Uji (fiktif)", "usia": "45",
            "kontak": "-", "gejala": "Tidak ada gejala"}
    if baris is not None:
        data.update({f: str(baris[f]) for f in FEATURES})
    data.update(ubah or {})
    for f in kosongkan or []:
        data[f] = ""
    return client.post("/", data=data).get_data(as_text=True)


hasil = []

# 1. Halaman utama
r = client.get("/")
hasil.append(("1", "Halaman skrining tampil", r.status_code == 200))

# 2. Sampel jinak
b = sampel[sampel.diagnosis == "B"].iloc[0]
html = kirim(b)
hasil.append(("2", "Sampel jinak -> Cenderung jinak", "Cenderung jinak (benign)" in html and "Cenderung ganas" not in html))

# 3. Sampel ganas
m = sampel[sampel.diagnosis == "M"].iloc[0]
html = kirim(m)
hasil.append(("3", "Sampel ganas -> Cenderung ganas", "Cenderung ganas (malignant)" in html and "Cenderung jinak" not in html))

# 4. Kolom kosong
html = kirim(m, kosongkan=["mean_radius", "worst_area"])
hasil.append(("4", "Kolom kosong -> pesan validasi",
              "belum diisi" in html))

# 5. Huruf dan nilai negatif
h1 = kirim(m, ubah={"mean_radius": "abc"})
h2 = kirim(m, ubah={"mean_radius": "-5"})
hasil.append(("5", "Huruf/negatif -> ditolak",
              "harus berupa angka" in h1 and "Cenderung ganas (malignant)" not in h2))

for no, ket, ok in hasil:
    print(f"Skenario {no}: {'LULUS' if ok else 'GAGAL'}  - {ket}")
sys.exit(0 if all(ok for *_, ok in hasil) else 1)
