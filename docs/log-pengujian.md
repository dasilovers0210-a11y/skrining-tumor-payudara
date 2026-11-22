# Log Pengujian

Pengujian dilakukan terhadap `app_flask.py` (Flask test client, skrip `tests/uji_aplikasi.py`) dengan data dari
`data_uji/sampel_uji.csv` (sampel diambil dari dataset Breast Cancer Wisconsin). Pengujian manual di browser
sebaiknya diulang oleh penyusun dan kolom "Hasil aktual" disesuaikan bila berbeda.

| No | Langkah | Hasil diharapkan | Hasil aktual | Status |
|---|---|---|---|---|
| 1 | Buka `http://127.0.0.1:5000/` | Halaman skrining tampil | Halaman tampil (HTTP 200) | Lulus |
| 2 | Isi 30 fitur dengan sampel jinak (id 8510426), kirim | Prediksi jinak | Kedua model menyatakan "Cenderung jinak (benign)" (LR 6,38%, DT 0,0% probabilitas ganas) | Lulus |
| 3 | Isi 30 fitur dengan sampel ganas (id 842302), kirim | Prediksi ganas | Kedua model menyatakan "Cenderung ganas (malignant)" (LR 100%, DT 100%) | Lulus |
| 4 | Kosongkan satu atau lebih kolom, kirim | Pesan validasi, tidak error | Muncul pesan "Nilai Mean - Radius belum diisi.", aplikasi tidak error | Lulus |
| 5 | Isi huruf (`abc`) atau nilai negatif (`-5`) pada kolom numerik, kirim | Ditolak dengan pesan yang jelas | Huruf: "harus berupa angka"; negatif: "harus berupa angka valid dan tidak negatif" | Lulus |

## Cara mengulang

```bash
python tests/uji_aplikasi.py
```
