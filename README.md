# Skrining Awal Tumor Payudara (Halaman Pasien)

Proyek Kapita Selekta (Prodi Teknik Komputer, Universitas Negeri Makassar, 2026):
*Dukungan Keputusan Skrining Tumor Payudara dengan Pendekatan AI Prediktif*.

Aplikasi web berbasis Flask untuk **skrining awal** tumor payudara. Dokter mengisi data pasien (fiktif) dan
30 fitur hasil pengukuran inti sel, lalu aplikasi menampilkan prediksi **jinak (benign)** atau **ganas
(malignant)**, probabilitas dari dua model, level risiko 1-5, dan fitur yang paling berpengaruh.

> **Peringatan:** aplikasi ini adalah alat bantu pembelajaran/skrining awal, **bukan diagnosis medis**.
> Hasil apa pun harus dikonfirmasi oleh tenaga medis.

## Tujuan

- Memberi dokter gambaran cepat dan dapat dijelaskan tentang kemungkinan tumor jinak atau ganas.
- Membandingkan dua model: Logistic Regression (model utama) dan Decision Tree (pembanding).
- Target awal: sensitivitas >= 95% dan spesifisitas >= 90% (ganas = positif).

## Struktur Folder

```
.
├── app_flask.py                      # Aplikasi web (Flask)
├── pelatihan.ipynb                   # Notebook pelatihan dan evaluasi model
├── halaman-pasien.html               # Rancangan halaman pasien (statis)
├── breast_cancer_wisconsin_diagnostic.csv   # Dataset
├── hasil_training/                   # Model (.joblib), metrik, feature importance, metadata
├── data_uji/
│   └── sampel_uji.csv                # 6 sampel (3 jinak, 3 ganas) untuk pengujian
├── tests/
│   └── uji_aplikasi.py               # Uji otomatis 5 skenario
├── docs/
│   └── log-pengujian.md              # Tabel skenario uji
├── requirements.txt
└── README.md
```

## Cara Menjalankan

Prasyarat: Python 3.10 atau lebih baru.

```bash
# 1. Salin repositori
git clone https://github.com/<dasilovers0210-a11y>/skrining-tumor-payudara.git
cd skrining-tumor-payudara

# 2. (Disarankan) buat virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Pasang dependensi
pip install -r requirements.txt

# 4. Jalankan aplikasi
python app_flask.py
```

Buka <http://127.0.0.1:5000/> di browser. Untuk mencoba cepat, gunakan tombol **Isi contoh jinak / ganas**
di halaman, atau salin nilai dari `data_uji/sampel_uji.csv`.

### Menjalankan pengujian

```bash
python tests/uji_aplikasi.py
```

### Melatih ulang model (opsional)

Buka `pelatihan.ipynb` dan jalankan semua sel. Model dan metrik disimpan ulang ke `hasil_training/`.

## Alat dan Versinya

| Alat | Versi / Keterangan |
|---|---|
| Python | 3.13 (diuji); 3.10+ seharusnya kompatibel |
| Flask | 3.1.x |
| scikit-learn | **1.9.0** (versi saat model dilatih; versi lain memunculkan peringatan versi) |
| pandas / numpy | 3.0.x / 2.5.x |
| matplotlib / seaborn / joblib | 3.11.x / 0.13.x / 1.6.x |
| Editor | Visual Studio Code |

> Cek versi yang Anda pakai dengan `pip freeze` dan sesuaikan tabel ini bila berbeda.

## Sumber Data

Breast Cancer Wisconsin (Diagnostic) Dataset: 569 sampel, 30 fitur numerik (mean, standard error, dan worst
dari 10 karakteristik inti sel), label `B` (benign) / `M` (malignant).
Sumber asli: UCI Machine Learning Repository
(<https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>).
Penyusun asli: W. H. Wolberg, W. N. Street, O. L. Mangasarian.
Data ini bukan data pasien nyata.

## Hasil Model (data uji 20%, `random_state=42`)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.965 | 0.975 | 0.929 | 0.951 | 0.996 |
| Decision Tree | 0.930 | 0.905 | 0.905 | 0.905 | 0.925 |

Cross-validation (rata-rata akurasi): Logistic Regression 0.974 ± 0.015, Decision Tree 0.923 ± 0.033.

Sensitivitas pada ambang 0,5: Logistic Regression 92,9% (39/42), Decision Tree 90,5% (38/42), jadi belum
mencapai target 95%. Ambang 0,3 memberi 97,6% (41/42) tetapi belum konsisten pada pembagian data lain,
sehingga masih hasil awal. Spesifisitas: 98,6% (LR) dan 94,4% (DT), target tercapai.

## Penyusun

- Dinda Ayu Miriam Alamsyah (240210500036)
- Reynaldi Exelsior Saputra (240210500022)
- Syahwal (240210501018)
- Wahyuni (240210501025)

Mata kuliah: Kapita Selekta (B10CKP02405), Kelompok 9, Kelas C3.
Dosen pengampu: Auliyah Zakilah Ifani, S.Kom., M.Kom.
