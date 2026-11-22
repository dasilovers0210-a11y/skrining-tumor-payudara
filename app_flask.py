from pathlib import Path
import base64
import math

import joblib
import pandas as pd
from flask import Flask, Response, abort, request, render_template_string

# =========================================================
# KONFIGURASI DAN LOAD MODEL
# =========================================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "hasil_training"

FEATURES = [
    "mean_radius", "mean_texture", "mean_perimeter", "mean_area",
    "mean_smoothness", "mean_compactness", "mean_concavity",
    "mean_concave_points", "mean_symmetry", "mean_fractal_dimension",
    "se_radius", "se_texture", "se_perimeter", "se_area",
    "se_smoothness", "se_compactness", "se_concavity",
    "se_concave_points", "se_symmetry", "se_fractal_dimension",
    "worst_radius", "worst_texture", "worst_perimeter", "worst_area",
    "worst_smoothness", "worst_compactness", "worst_concavity",
    "worst_concave_points", "worst_symmetry", "worst_fractal_dimension",
]

FEATURE_LABELS = {
    "radius": "Radius",
    "texture": "Tekstur",
    "perimeter": "Keliling",
    "area": "Luas",
    "smoothness": "Kehalusan",
    "compactness": "Kekompakan",
    "concavity": "Kecekungan",
    "concave_points": "Titik cekung",
    "symmetry": "Simetri",
    "fractal_dimension": "Dimensi fraktal",
}

FEATURE_DESCRIPTIONS = {
    "radius": "Rata-rata jarak dari pusat ke tepi inti sel.",
    "texture": "Variasi tingkat keabuan pada citra sel.",
    "perimeter": "Panjang tepi inti sel.",
    "area": "Luas inti sel.",
    "smoothness": "Variasi panjang garis tepi sel.",
    "compactness": "Ukuran kekompakan bentuk inti sel.",
    "concavity": "Tingkat cekungan pada tepi sel.",
    "concave_points": "Jumlah bagian tepi yang cekung.",
    "symmetry": "Kesimetrisan bentuk inti sel.",
    "fractal_dimension": "Kerumitan garis tepi sel.",
}

# Rentang umum dataset WDBC.
# Rentang ini hanya panduan input, bukan batas klinis.

RANGES = {
    "mean": {
        "radius": (6.981, 28.11),
        "texture": (9.71, 39.28),
        "perimeter": (43.79, 188.5),
        "area": (143.5, 2501),
        "smoothness": (0.05263, 0.1634),
        "compactness": (0.01938, 0.3454),
        "concavity": (0.0, 0.4268),
        "concave_points": (0.0, 0.2012),
        "symmetry": (0.106, 0.304),
        "fractal_dimension": (0.04996, 0.09744),
    },
    "se": {
        "radius": (0.1115, 2.873),
        "texture": (0.3602, 4.885),
        "perimeter": (0.757, 21.98),
        "area": (6.802, 542.2),
        "smoothness": (0.001713, 0.03113),
        "compactness": (0.002252, 0.1354),
        "concavity": (0.0, 0.396),
        "concave_points": (0.0, 0.05279),
        "symmetry": (0.007882, 0.07895),
        "fractal_dimension": (0.000895, 0.02984),
    },
    "worst": {
        "radius": (7.93, 36.04),
        "texture": (12.02, 49.54),
        "perimeter": (50.41, 251.2),
        "area": (185.2, 4254),
        "smoothness": (0.07117, 0.2226),
        "compactness": (0.02729, 1.058),
        "concavity": (0.0, 1.252),
        "concave_points": (0.0, 0.291),
        "symmetry": (0.1565, 0.6638),
        "fractal_dimension": (0.05504, 0.2075),
    },
}

GROUPS = {
    "mean": {
        "title": "Rata-rata (Mean)",
        "short": "Mean",
        "description": "Pengukuran rata-rata karakteristik inti sel.",
        "features": FEATURES[:10],
    },
    "se": {
        "title": "Galat baku (SE)",
        "short": "SE",
        "description": "Pengukuran galat baku karakteristik inti sel.",
        "features": FEATURES[10:20],
    },
    "worst": {
        "title": "Terburuk (Worst)",
        "short": "Worst",
        "description": "Pengukuran nilai terburuk karakteristik inti sel.",
        "features": FEATURES[20:30],
    },
}

SYMPTOMS = [
    "Benjolan di payudara",
    "Nyeri pada payudara",
    "Perubahan kulit (kemerahan atau berkerut)",
    "Keluar cairan dari puting",
    "Puting tertarik ke dalam",
    "Benjolan di ketiak",
    "Tidak ada gejala",
]

RISK_COLORS = [
    "#2E8B57",
    "#7DAF3C",
    "#E0B21F",
    "#E5782A",
    "#C23B3B",
]


# =========================================================
# FUNGSI BANTUAN
# =========================================================

def get_feature_label(feature):

    prefix, name = feature.split("_", 1)

    prefix_label = {
        "mean": "Mean",
        "se": "SE",
        "worst": "Worst",
    }[prefix]

    return f"{prefix_label} - {FEATURE_LABELS.get(name, name)}"


def get_feature_step(feature):

    prefix, name = feature.split("_", 1)

    if name == "area":
        return 0.1 if prefix == "se" else 1

    if name in {"radius", "texture", "perimeter"}:
        return 0.001 if prefix == "se" else 0.01

    return 0.0001


def load_model(path):

    if not path.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {path}"
        )

    model = joblib.load(path)

    if (
        not hasattr(model, "predict_proba")
        or not hasattr(model, "classes_")
    ):
        raise ValueError(
            f"Model {path.name} harus mendukung predict_proba() "
            "dan memiliki classes_."
        )

    return model


# Memuat model hasil training

logistic_model = load_model(
    MODEL_DIR / "logistic_regression.joblib"
)

decision_tree_model = load_model(
    MODEL_DIR / "decision_tree.joblib"
)


def check_model_features(model, model_name):

    trained_features = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if trained_features is not None:

        if list(trained_features) != FEATURES:

            raise ValueError(
                f"Nama/urutan fitur {model_name} tidak sama "
                "dengan FEATURES aplikasi. Periksa urutan "
                "fitur saat training."
            )


check_model_features(
    logistic_model,
    "Logistic Regression",
)

check_model_features(
    decision_tree_model,
    "Decision Tree",
)


def malignant_probability(model, input_data):

    probabilities = model.predict_proba(input_data)[0]

    malignant_index = None

    for index, class_value in enumerate(model.classes_):

        label = str(class_value).strip().upper()

        if label in {"1", "M", "MALIGNANT"}:
            malignant_index = index
            break

    if malignant_index is None:

        raise ValueError(
            "Kelas Malignant tidak ditemukan pada model. "
            f"Kelas tersedia: {list(model.classes_)}. "
            "Periksa label target saat training."
        )

    probability = float(
        probabilities[malignant_index]
    )

    if (
        not math.isfinite(probability)
        or not 0 <= probability <= 1
    ):

        raise ValueError(
            "Model menghasilkan probabilitas yang tidak valid."
        )

    return probability


# =========================================================
# LEVEL PROBABILITAS
# =========================================================

def get_level(probability):

    if probability < 0.10:
        return 1, "Sangat rendah", (
            "Korelasikan hasil model dengan pemeriksaan klinis, "
            "pencitraan, dan sitologi. Hasil rendah tidak otomatis "
            "menyingkirkan keganasan."
        )

    if probability < 0.30:
        return 2, "Rendah", (
            "Tinjau kesesuaian prediksi dengan temuan pemeriksaan "
            "pasien dan tentukan tindak lanjut berdasarkan pedoman klinis."
        )

    if probability < 0.50:
        return 3, "Sedang", (
            "Pertimbangkan penelaahan lebih lanjut jika terdapat "
            "ketidaksesuaian hasil pemeriksaan."
        )

    if probability < 0.80:
        return 4, "Tinggi", (
            "Tinjau hasil sitologi bersama temuan klinis dan pencitraan "
            "serta pertimbangkan evaluasi lanjutan."
        )

    return 5, "Sangat tinggi", (
        "Prioritaskan penelaahan hasil dan tentukan kebutuhan "
        "pemeriksaan lanjutan sesuai kondisi pasien."
    )


def predict_model(model, input_data):

    probability = malignant_probability(
        model,
        input_data,
    )

    classification = (
        "Malignant"
        if probability >= 0.5
        else "Benign"
    )

    label = (
        "Cenderung ganas (malignant)"
        if probability >= 0.5
        else "Cenderung jinak (benign)"
    )

    level_number, level_label, advice = get_level(
        probability
    )

    return {
        "probability": round(probability * 100, 2),
        "raw_probability": probability,
        "classification": classification,
        "label": label,
        "level_number": level_number,
        "level_label": level_label,
        "advice": advice,
    }


# =========================================================
# FITUR YANG PALING BERPENGARUH
# =========================================================

def get_feature_contributions(model, input_data):

    """
    Mengambil kontribusi fitur dari koefisien Logistic Regression.

    Mendukung:
    - Logistic Regression biasa.
    - Pipeline yang memiliki transformer dan Logistic Regression
      sebagai estimator terakhir.

    Jika model tidak memiliki koefisien yang sesuai, fungsi
    mengembalikan daftar kosong. Tidak menggunakan angka simulasi.
    """

    estimator = model
    transformed_data = input_data

    # Jika model merupakan Pipeline sklearn

    if hasattr(model, "steps"):

        try:

            for _, transformer in model.steps[:-1]:

                transformed_data = transformer.transform(
                    transformed_data
                )

            estimator = model.steps[-1][1]

        except Exception:

            return []

    if not hasattr(estimator, "coef_"):
        return []

    try:

        coefficients = estimator.coef_[0]

        if hasattr(transformed_data, "toarray"):
            values = transformed_data.toarray()[0]

        elif hasattr(transformed_data, "iloc"):
            values = transformed_data.iloc[0].to_numpy()

        else:
            values = transformed_data[0]

        if len(coefficients) != len(FEATURES):
            return []

        if len(values) != len(FEATURES):
            return []

        contributions = []

        for index, feature in enumerate(FEATURES):

            contribution = (
                float(coefficients[index])
                * float(values[index])
            )

            contributions.append({
                "feature": feature,
                "label": get_feature_label(feature),
                "value": contribution,
                "abs_value": abs(contribution),
                "direction": (
                    "naik"
                    if contribution > 0
                    else "turun"
                ),
            })

        contributions.sort(
            key=lambda item: item["abs_value"],
            reverse=True,
        )

        return contributions[:5]

    except Exception:

        app.logger.exception(
            "Gagal menghitung kontribusi fitur."
        )

        return []


# =========================================================
# HTML DAN CSS
# =========================================================

HTML = r"""
<!DOCTYPE html>
<html lang="id">

<head>

<script>
(function(){
  var t='light';
  try{ if(localStorage.getItem('tema')==='dark'){ t='dark'; } }catch(e){}
  document.documentElement.setAttribute('data-theme',t);
})();
</script>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1, viewport-fit=cover">

<title>Skrining Awal Tumor Payudara</title>

<style>

:root{
--bg:#F2F6F7;
--card:#fff;
--ink:#14262B;
--mute:#55696E;
--line:#D5E0E2;
--accent:#0F6B74;
--accent-d:#0A4F56;
--warn-bg:#FFF4DC;
--warn-ink:#7A4A00;
--r1:#2E8B57;
--r2:#7DAF3C;
--r3:#E0B21F;
--r4:#E5782A;
--r5:#C23B3B;
}

*{
box-sizing:border-box;
}

body{
margin:0;
background:var(--bg);
color:var(--ink);
font:16px/1.6 "Segoe UI",system-ui,-apple-system,sans-serif;
}

h1,h2,h3{
font-family:Georgia,"Times New Roman",serif;
line-height:1.25;
margin:0 0 .5em;
}

h1{
font-size:2.1rem;
}

h2{
font-size:1.5rem;
}

h3{
font-size:1.1rem;
}

.notice{
position:sticky;
top:0;
z-index:10;
background:var(--warn-bg);
color:var(--warn-ink);
border-bottom:1px solid #EBCB8B;
padding:.55rem 1rem;
font-size:.9rem;
text-align:center;
}

.wrap{
max-width:860px;
margin:0 auto;
padding:1.5rem 1rem 3rem;
}

.brand{
font-family:Georgia,serif;
font-weight:700;
color:var(--accent-d);
padding:1rem 0;
}

.card{
background:var(--card);
border:1px solid var(--line);
border-radius:10px;
padding:1.5rem;
margin-bottom:1rem;
}

.view{
display:none;
}

.view.on{
display:block;
}

.steps{
display:flex;
gap:.5rem;
margin-bottom:1rem;
font-size:.9rem;
color:var(--mute);
}

.steps span{
flex:1;
border-top:4px solid var(--line);
padding-top:.35rem;
}

.steps span.done{
border-color:var(--accent);
color:var(--ink);
font-weight:600;
}

.btn{
display:inline-block;
border:0;
border-radius:8px;
padding:.7rem 1.3rem;
font:inherit;
font-weight:600;
cursor:pointer;
background:var(--accent);
color:#fff;
text-decoration:none;
}

.btn:hover{
background:var(--accent-d);
}

.btn.sec{
background:var(--card);
color:var(--accent-d);
border:1px solid var(--accent);
}

.btn.sec:hover{
background:#EAF3F4;
}

.btn:disabled{
opacity:.6;
cursor:wait;
}

.row{
display:flex;
gap:.75rem;
flex-wrap:wrap;
margin-top:1.25rem;
}

:focus-visible{
outline:3px solid #2B8CFF;
outline-offset:2px;
}

label{
display:block;
font-weight:600;
margin-bottom:.25rem;
}

.opt{
font-weight:400;
color:var(--mute);
}

input[type=text],
input[type=number],
input[type=tel]{
width:100%;
padding:.6rem .7rem;
border:1px solid #AFC0C3;
border-radius:6px;
font:inherit;
background:#fff;
color:#14262B;
}

/* Panah bawaan input angka sengaja dipertahankan. */


input.bad{
border-color:#C23B3B;
background:#FFF2F2;
}

input.out{
border-color:#E5A52A;
background:#FFFAEB;
}

.grid{
display:grid;
grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
gap:1rem;
}

.hint{
font-size:.8rem;
color:var(--mute);
margin-top:.2rem;
}

.err{
background:#FFF0F0;
color:#A32121;
border:1px solid #E9BABA;
border-radius:7px;
padding:.7rem .9rem;
margin:.75rem 0;
}

.sym{
display:grid;
grid-template-columns:repeat(auto-fit,minmax(240px,1fr));
gap:.4rem .8rem;
margin:.5rem 0;
}

.sym label{
font-weight:400;
display:flex;
gap:.5rem;
align-items:center;
margin:0;
}

.sym input{
width:17px;
height:17px;
}

.note{
background:#EAF3F4;
border-left:4px solid var(--accent);
padding:.6rem .9rem;
border-radius:4px;
font-size:.92rem;
margin:.75rem 0;
}

.guide{
display:grid;
grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
gap:1rem;
margin:1.5rem 0;
}

.guide div{
border-left:3px solid var(--accent);
padding-left:.8rem;
}

.tabs{
display:flex;
gap:.4rem;
margin:1rem 0;
flex-wrap:wrap;
}

.tab{
border:1px solid var(--line);
background:var(--card);
color:var(--ink);
border-radius:8px 8px 0 0;
padding:.5rem 1rem;
font:inherit;
cursor:pointer;
}

.tab[aria-selected=true]{
background:var(--accent);
color:#fff;
border-color:var(--accent);
}

.pane{
display:none;
}

.pane.on{
display:block;
}

/* HASIL SKRINING MENGIKUTI HTML TEMAN */

.res-top{
display:flex;
gap:1.5rem;
flex-wrap:wrap;
align-items:center;
}

.big{
font-family:Georgia,serif;
font-size:2.6rem;
line-height:1;
}

.gauge{
display:grid;
grid-template-columns:repeat(5,1fr);
gap:4px;
margin:1rem 0 .3rem;
}

.gauge div{
height:16px;
border-radius:3px;
opacity:.28;
}

.gauge div.on{
opacity:1;
outline:3px solid var(--ink);
outline-offset:1px;
}

.glabels{
display:grid;
grid-template-columns:repeat(5,1fr);
gap:4px;
font-size:.75rem;
color:var(--mute);
text-align:center;
}

.bar{
display:grid;
grid-template-columns:190px 1fr 50px;
gap:.6rem;
align-items:center;
font-size:.9rem;
margin:.45rem 0;
}

.bar i{
display:block;
height:12px;
background:var(--accent);
border-radius:3px;
}

.sum dt{
font-weight:600;
float:left;
clear:left;
width:150px;
}

.sum dd{
margin:0 0 .3rem 150px;
overflow-wrap:anywhere;
}

.model-comparison{
display:grid;
grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
gap:1rem;
margin:1rem 0;
}

.model-box{
background:#EAF3F4;
padding:1rem;
border-radius:8px;
}

.model-box .pct{
font:700 1.7rem Georgia,serif;
margin:.3rem 0;
}

footer{
margin-top:2rem;
font-size:.8rem;
color:var(--mute);
text-align:center;
}

@media(max-width:600px){

h1{
font-size:1.6rem;
}

.bar{
grid-template-columns:110px 1fr 44px;
}

.sum dt{
float:none;
width:auto;
}

.sum dd{
margin-left:0;
}

.steps{
font-size:.75rem;
}

.card{
padding:1.1rem;
}


}

@media print{

.notice,
.row,
.steps,
.tabs,
footer{
display:none!important;
}

body{
background:white;
color:black;
}

.card{
border:0;
}

}

:root{
padding-top:env(safe-area-inset-top,0px);
padding-bottom:env(safe-area-inset-bottom,0px);
}

html{
scroll-padding-top:env(safe-area-inset-top,0px);
}

/* Mode gelap kini diatur lewat tombol tema (lihat lapisan tampilan di bawah). */

/* =====================================================
   LAPISAN TAMPILAN RUMAH SAKIT 3D
   (hanya visual, tidak mengubah alur aplikasi)
===================================================== */

:root{
--font:"Segoe UI","Inter",system-ui,-apple-system,"Helvetica Neue",Arial,sans-serif;

--bg:#EAF3F9;
--card:#FFFFFF;
--ink:#12304A;
--mute:#57718A;
--line:#D2E2EE;
--accent:#0A7EA4;
--accent-d:#075B7A;
--warn-bg:#FFF4DC;
--warn-ink:#7A4A00;

--bg-a:#DCEBF5;
--bg-b:#F6FAFD;
--card-hi:#FFFFFF;
--card-lo:#F1F7FB;
--tile-hi:#FFFFFF;
--tile-lo:#E9F2F8;
--field-a:#E9F1F7;
--field-b:#FFFFFF;
--hi:rgba(255,255,255,.95);
--sh-1:rgba(14,60,100,.10);
--sh-2:rgba(14,60,100,.22);
--inset:rgba(14,60,100,.16);
--glow:rgba(10,126,164,.38);
--btn-top:#2BAAD1;
--btn-bot:#0A78A0;
--btn-edge:#064F6B;
--strip:linear-gradient(90deg,#0A7EA4,#1AA5B0,#3BC596);
--cross-line:rgba(10,126,164,.08);
}

html[data-theme="dark"]{
color-scheme:dark;
--bg:#0A1620;
--card:#13263A;
--ink:#E8F1F8;
--mute:#9FB6C9;
--line:#27425C;
--accent:#4DB8E0;
--accent-d:#8FD6F0;
--warn-bg:#3A2D10;
--warn-ink:#F2D08A;

--bg-a:#08121A;
--bg-b:#10243A;
--card-hi:#1B3350;
--card-lo:#122339;
--tile-hi:#1F3A58;
--tile-lo:#142A42;
--field-a:#0B1826;
--field-b:#12283D;
--hi:rgba(255,255,255,.08);
--sh-1:rgba(0,0,0,.35);
--sh-2:rgba(0,0,0,.62);
--inset:rgba(0,0,0,.55);
--glow:rgba(77,184,224,.40);
--btn-top:#2F9FCB;
--btn-bot:#14719A;
--btn-edge:#0A4560;
--cross-line:rgba(143,214,240,.05);
}

/* ---------- Dasar halaman ---------- */

html{
min-height:100%;
}

body{
font-family:var(--font);
min-height:100vh;
background:
url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='64' height='64'%3E%3Cpath d='M32 24v16M24 32h16' stroke='%230A7EA4' stroke-opacity='.10' stroke-width='3.5' stroke-linecap='round'/%3E%3C/svg%3E"),
radial-gradient(900px 520px at 6% -8%, rgba(43,170,209,.25), transparent 60%),
radial-gradient(800px 500px at 100% 0%, rgba(59,197,150,.16), transparent 55%),
linear-gradient(180deg,var(--bg-a),var(--bg-b)) fixed;
background-attachment:scroll,fixed,fixed,fixed;
}

html[data-theme="dark"] body{
background:
url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='64' height='64'%3E%3Cpath d='M32 24v16M24 32h16' stroke='%238FD6F0' stroke-opacity='.06' stroke-width='3.5' stroke-linecap='round'/%3E%3C/svg%3E"),
radial-gradient(900px 520px at 6% -8%, rgba(47,159,203,.22), transparent 60%),
radial-gradient(800px 500px at 100% 0%, rgba(59,197,150,.10), transparent 55%),
linear-gradient(180deg,var(--bg-a),var(--bg-b)) fixed;
background-attachment:scroll,fixed,fixed,fixed;
}

h1,h2,h3,.brand,.big,.model-box .pct{
font-family:var(--font);
}

h1,h2,h3{
font-weight:700;
letter-spacing:-.01em;
}

.wrap{
max-width:920px;
padding:1.2rem 1rem 3rem;
}

/* ---------- Header / brand ---------- */

.brand{
display:flex;
align-items:center;
gap:.85rem;
margin:.4rem 0 1.3rem;
padding:.8rem 1.1rem;
border:1px solid var(--line);
border-radius:20px;
font-size:1.2rem;
font-weight:700;
letter-spacing:.1px;
color:var(--accent-d);
background:
url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 260 60'%3E%3Cpath d='M0 32h60l10-6 8 6h18l8-26 12 50 10-24h20l8-8 8 8h98' fill='none' stroke='%230A7EA4' stroke-opacity='.35' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E") no-repeat right 150px center/170px 34px,
linear-gradient(160deg,var(--card-hi),var(--card-lo));
box-shadow:
inset 0 1px 0 var(--hi),
0 3px 0 var(--line),
0 16px 26px -14px var(--sh-2);
}

.brand::before{
content:"";
flex:0 0 auto;
width:46px;
height:46px;
border-radius:14px;
background:
linear-gradient(#fff,#fff) center/54% 16% no-repeat,
linear-gradient(#fff,#fff) center/16% 54% no-repeat,
linear-gradient(160deg,var(--btn-top),var(--btn-bot));
box-shadow:
inset 0 2px 0 rgba(255,255,255,.4),
inset 0 -4px 0 rgba(0,0,0,.16),
0 4px 0 var(--btn-edge),
0 10px 18px -2px var(--glow);
transform:none;
}

.brand-title{
flex:1 1 auto;
min-width:0;
}

.theme-toggle{
flex:0 0 auto;
margin-left:auto;
font:600 .8rem var(--font);
color:var(--accent-d);
padding:.5rem .95rem;
border:1px solid var(--accent);
border-radius:999px;
background:linear-gradient(180deg,var(--tile-hi),var(--tile-lo));
box-shadow:inset 0 1px 0 var(--hi),0 3px 0 var(--line),0 8px 12px -6px var(--sh-2);
cursor:pointer;
transition:transform .15s,box-shadow .15s;
}

.theme-toggle:hover{
transform:translateY(-2px);
}

.theme-toggle:active{
transform:translateY(2px);
box-shadow:inset 0 1px 0 var(--hi),0 1px 0 var(--line);
}

/* ---------- Kartu ---------- */

.card{
position:relative;
background:linear-gradient(160deg,var(--card-hi),var(--card-lo));
border:1px solid var(--line);
border-radius:22px;
padding:2rem;
box-shadow:
inset 0 1px 0 var(--hi),
0 2px 4px var(--sh-1),
0 16px 26px -8px var(--sh-1),
0 44px 70px -28px var(--sh-2);
}

.card::before{
content:"";
position:absolute;
left:26px;
right:26px;
top:0;
height:5px;
border-radius:0 0 8px 8px;
background:var(--strip);
}

#result::before{
background:linear-gradient(90deg,var(--r1),var(--r2),var(--r3),var(--r4),var(--r5));
}

h1{
font-size:2rem;
background:linear-gradient(120deg,var(--accent-d),var(--accent) 70%,#2FB890);
-webkit-background-clip:text;
background-clip:text;
-webkit-text-fill-color:transparent;
color:var(--accent-d);
}

h2{
display:inline-block;
padding-bottom:.3rem;
margin-bottom:.9rem;
border-bottom:4px solid var(--accent);
border-radius:0 0 3px 3px;
}

#v1 .card{
background:
url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 260 60'%3E%3Cpath d='M0 32h60l10-6 8 6h18l8-26 12 50 10-24h20l8-8 8 8h98' fill='none' stroke='%230A7EA4' stroke-opacity='.22' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E") no-repeat right 1.6rem bottom 1.4rem/280px 64px,
linear-gradient(160deg,var(--card-hi),var(--card-lo));
}

/* ---------- Animasi pergantian halaman ---------- */

.view.on{
animation:viewIn .5s cubic-bezier(.2,.8,.2,1) both;
}

.pane.on{
animation:viewIn .35s ease both;
}

@keyframes viewIn{
from{opacity:0;transform:translateY(16px);}
to{opacity:1;transform:none;}
}

/* ---------- Langkah 1 - 2 - 3 (masuk ke dalam kolom) ---------- */

.steps{
gap:.7rem;
margin:0 .7rem 1.2rem;
padding:0;
}

.steps span{
flex:1;
min-width:0;
padding:.7rem 1rem;
border:1px solid var(--line);
border-radius:14px;
text-align:center;
font-weight:600;
color:var(--mute);
background:linear-gradient(180deg,var(--tile-hi),var(--tile-lo));
box-shadow:
inset 0 1px 0 var(--hi),
0 3px 0 var(--line),
0 12px 16px -12px var(--sh-2);
white-space:nowrap;
overflow:hidden;
text-overflow:ellipsis;
}

.steps span.done{
color:#fff;
text-shadow:0 1px 1px rgba(0,0,0,.28);
border-color:transparent;
background:linear-gradient(180deg,var(--btn-top),var(--btn-bot));
box-shadow:
inset 0 1px 0 rgba(255,255,255,.4),
0 4px 0 var(--btn-edge),
0 14px 18px -8px var(--glow);
font-weight:700;
}

/* ---------- Tombol 3D ---------- */

.btn{
position:relative;
border-radius:14px;
padding:.8rem 1.6rem;
background:linear-gradient(180deg,var(--btn-top),var(--btn-bot));
color:#fff;
text-shadow:0 1px 1px rgba(0,0,0,.3);
box-shadow:
inset 0 1px 0 rgba(255,255,255,.4),
0 5px 0 var(--btn-edge),
0 12px 18px -4px var(--glow);
transform:translateY(0);
transition:transform .15s ease,box-shadow .15s ease,filter .15s ease;
}

.btn:hover{
background:linear-gradient(180deg,var(--btn-top),var(--btn-bot));
filter:brightness(1.07);
transform:translateY(-2px);
box-shadow:
inset 0 1px 0 rgba(255,255,255,.4),
0 7px 0 var(--btn-edge),
0 18px 24px -6px var(--glow);
}

.btn:active{
transform:translateY(4px);
box-shadow:
inset 0 1px 0 rgba(255,255,255,.3),
0 1px 0 var(--btn-edge),
0 4px 8px -2px var(--glow);
}

.btn.sec{
background:linear-gradient(180deg,var(--tile-hi),var(--tile-lo));
color:var(--accent-d);
text-shadow:none;
border:1px solid var(--accent);
box-shadow:
inset 0 1px 0 var(--hi),
0 4px 0 var(--line),
0 10px 16px -6px var(--sh-2);
}

.btn.sec:hover{
background:linear-gradient(180deg,var(--tile-hi),var(--tile-lo));
filter:brightness(1.04);
transform:translateY(-2px);
box-shadow:
inset 0 1px 0 var(--hi),
0 6px 0 var(--line),
0 16px 20px -8px var(--sh-2);
}

.btn.sec:active{
transform:translateY(3px);
box-shadow:inset 0 1px 0 var(--hi),0 1px 0 var(--line);
}

.btn:disabled,
.btn:disabled:hover,
.btn:disabled:active{
transform:translateY(3px);
filter:grayscale(.3);
box-shadow:0 2px 0 var(--btn-edge);
}

.row{
gap:1rem;
padding-bottom:.4rem;
}

/* ---------- Input ---------- */

input[type=text],
input[type=number],
input[type=tel]{
border-radius:12px;
border:1px solid var(--line);
padding:.7rem .85rem;
color:var(--ink);
background:linear-gradient(180deg,var(--field-a),var(--field-b));
box-shadow:inset 0 3px 6px var(--inset),0 1px 0 var(--hi);
transition:box-shadow .2s,border-color .2s;
}

input[type=text]:focus,
input[type=number]:focus,
input[type=tel]:focus{
border-color:var(--accent);
box-shadow:inset 0 2px 4px var(--inset),0 0 0 4px var(--glow);
}

input.bad{
border-color:#C23B3B;
background:linear-gradient(180deg,#FFE6E6,#FFF4F4);
box-shadow:inset 0 3px 6px var(--inset),0 0 0 3px rgba(194,59,59,.28);
}

input.out{
border-color:#E5A52A;
background:linear-gradient(180deg,#FFF1CF,#FFFAEB);
box-shadow:inset 0 3px 6px var(--inset),0 0 0 3px rgba(229,165,42,.28);
}

html[data-theme="dark"] input.bad{
background:#3A1717;
}

html[data-theme="dark"] input.out{
background:#3A2D10;
}

/* ---------- Catatan / peringatan ---------- */

.note{
border-radius:14px;
border-left-width:6px;
padding:.85rem 1.1rem;
background:linear-gradient(135deg,var(--tile-lo),var(--tile-hi));
box-shadow:inset 0 1px 0 var(--hi),0 8px 16px -8px var(--sh-2);
}

.err{
border-radius:14px;
box-shadow:0 8px 18px -8px rgba(194,59,59,.45);
}

/* ---------- Panduan beranda ---------- */

.guide{
gap:1.2rem;
perspective:900px;
}

.guide div{
border:1px solid var(--line);
border-left:5px solid var(--accent);
border-radius:16px;
padding:1.1rem 1.1rem 1.1rem 1.2rem;
background:linear-gradient(160deg,var(--tile-hi),var(--tile-lo));
box-shadow:
inset 0 1px 0 var(--hi),
0 3px 0 var(--line),
0 14px 22px -10px var(--sh-2);
transition:transform .25s ease,box-shadow .25s ease;
}

.guide div:hover{
transform:translateY(-5px);
box-shadow:
inset 0 1px 0 var(--hi),
0 3px 0 var(--line),
0 26px 32px -12px var(--sh-2);
}

/* ---------- Gejala ---------- */

.sym label{
padding:.65rem .9rem;
border:1px solid var(--line);
border-radius:14px;
background:linear-gradient(160deg,var(--tile-hi),var(--tile-lo));
box-shadow:inset 0 1px 0 var(--hi),0 3px 0 var(--line),0 8px 14px -8px var(--sh-2);
cursor:pointer;
transition:transform .15s,box-shadow .15s,border-color .15s;
}

.sym label:hover{
transform:translateY(-2px);
}

.sym label:has(input:checked){
border-color:var(--accent);
box-shadow:inset 0 2px 5px var(--inset),0 0 0 2px var(--accent);
transform:translateY(2px);
}

.sym input{
accent-color:var(--accent);
}

/* ---------- Tab ---------- */

.tabs{
gap:.6rem;
}

.tab{
border-radius:14px;
padding:.65rem 1.2rem;
font-weight:600;
background:linear-gradient(180deg,var(--tile-hi),var(--tile-lo));
box-shadow:inset 0 1px 0 var(--hi),0 4px 0 var(--line),0 10px 14px -8px var(--sh-2);
transition:transform .15s,box-shadow .15s;
}

.tab:hover{
transform:translateY(-2px);
}

.tab[aria-selected=true]{
background:linear-gradient(180deg,var(--btn-top),var(--btn-bot));
color:#fff;
text-shadow:0 1px 1px rgba(0,0,0,.3);
transform:translateY(-3px);
box-shadow:inset 0 1px 0 rgba(255,255,255,.4),0 6px 0 var(--btn-edge),0 16px 22px -8px var(--glow);
}

/* ---------- Hasil skrining ---------- */

.big{
font-weight:800;
background:linear-gradient(135deg,var(--accent-d),var(--accent));
-webkit-background-clip:text;
background-clip:text;
-webkit-text-fill-color:transparent;
color:var(--accent-d);
filter:drop-shadow(0 6px 8px var(--sh-2));
}

.res-top{
padding:1.1rem 1.4rem;
border-radius:18px;
border:1px solid var(--line);
background:linear-gradient(160deg,var(--tile-hi),var(--tile-lo));
box-shadow:inset 0 1px 0 var(--hi),0 4px 0 var(--line),0 18px 26px -14px var(--sh-2);
}

.gauge{
gap:6px;
margin:1.2rem 0 .5rem;
}

.gauge div{
position:relative;
height:22px;
border-radius:9px;
opacity:.35;
box-shadow:inset 0 -4px 0 rgba(0,0,0,.2),0 3px 0 rgba(0,0,0,.15);
transition:transform .3s;
}

.gauge div::after{
content:"";
position:absolute;
left:0;
right:0;
top:0;
height:50%;
border-radius:9px 9px 0 0;
background:linear-gradient(rgba(255,255,255,.55),rgba(255,255,255,0));
}

.gauge div.on{
opacity:1;
transform:translateY(-4px) scale(1.08);
box-shadow:inset 0 -4px 0 rgba(0,0,0,.2),0 8px 0 rgba(0,0,0,.18),0 14px 20px -4px rgba(0,0,0,.35);
}

.bar i{
height:14px;
border-radius:9px;
background:linear-gradient(180deg,var(--btn-top),var(--btn-bot));
box-shadow:inset 0 2px 0 rgba(255,255,255,.4),inset 0 -3px 0 rgba(0,0,0,.2),0 4px 8px -2px var(--sh-2);
}

.model-comparison{
gap:1.3rem;
perspective:1000px;
}

.model-box{
border:1px solid var(--line);
border-radius:18px;
padding:1.2rem;
background:linear-gradient(160deg,var(--tile-hi),var(--tile-lo));
box-shadow:inset 0 1px 0 var(--hi),0 5px 0 var(--line),0 22px 30px -14px var(--sh-2);
transition:transform .3s ease,box-shadow .3s ease;
}

.model-box:hover{
transform:translateY(-6px);
box-shadow:inset 0 1px 0 var(--hi),0 5px 0 var(--line),0 32px 38px -14px var(--sh-2);
}

.model-box .pct{
font-weight:800;
color:var(--accent-d);
}

.sum{
padding:1rem 1.2rem;
border:1px solid var(--line);
border-radius:16px;
background:linear-gradient(160deg,var(--tile-hi),var(--tile-lo));
box-shadow:inset 0 1px 0 var(--hi),0 10px 18px -12px var(--sh-2);
overflow:hidden;
}

footer{
padding-top:1rem;
}

/* ---------- Layar kecil & aksesibilitas ---------- */

@media(max-width:700px){

.brand{
font-size:1rem;
padding:.65rem .8rem;
background:linear-gradient(160deg,var(--card-hi),var(--card-lo));
}

.brand::before{
width:38px;
height:38px;
}

.card{
padding:1.2rem;
border-radius:18px;
}

.steps{
margin:0 .2rem 1rem;
gap:.4rem;
}

.steps span{
padding:.55rem .4rem;
font-size:.72rem;
}

#v1 .card{
background-image:linear-gradient(160deg,var(--card-hi),var(--card-lo));
}

.guide div:hover{
transform:none;
}

}

@media(prefers-reduced-motion:reduce){

*,
*::before,
*::after{
animation:none!important;
transition:none!important;
}

}

/* ---------- Cetak: tanpa efek 3D ---------- */

@media print{

body{
background:#fff!important;
}

.brand,
.card,
.model-box,
.res-top,
.note,
.sum{
box-shadow:none!important;
transform:none!important;
background:#fff!important;
}

.theme-toggle,
.card::before,
.brand::before{
display:none!important;
}

h1,
.big{
-webkit-text-fill-color:currentColor;
color:#000;
}

}


/* =====================================================
   TAMBAHAN: latar foto dokter per halaman + perbaikan cetak
   (hanya visual, tidak mengubah alur aplikasi)
===================================================== */

:root{
--card-a:rgba(255,255,255,.88);
--card-b:rgba(241,247,251,.86);
}

html[data-theme="dark"]{
--card-a:rgba(27,51,80,.88);
--card-b:rgba(18,35,57,.86);
}

body::before{
content:"";
position:fixed;
top:0;
right:0;
bottom:0;
left:0;
z-index:-1;
pointer-events:none;
background-position:center;
background-size:cover;
background-repeat:no-repeat;
opacity:.34;
transition:opacity .4s;
}

html[data-theme="dark"] body::before{
opacity:.22;
filter:brightness(.8);
}

body:has(#v1.on)::before{ background-image:url("/bg/beranda"); }
body:has(#v2.on)::before{ background-image:url("/bg/data-pasien"); }
body:has(#v3.on)::before{ background-image:url("/bg/hasil-sitologi"); }
body:has(#v4.on)::before{ background-image:url("/bg/hasil"); }

.card{
background:linear-gradient(160deg,var(--card-a),var(--card-b));
-webkit-backdrop-filter:blur(5px);
backdrop-filter:blur(5px);
}

#v1 .card{
background:
url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 260 60'%3E%3Cpath d='M0 32h60l10-6 8 6h18l8-26 12 50 10-24h20l8-8 8 8h98' fill='none' stroke='%230A7EA4' stroke-opacity='.22' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E") no-repeat right 1.6rem bottom 1.4rem/280px 64px,
linear-gradient(160deg,var(--card-a),var(--card-b));
}

@media(max-width:700px){
#v1 .card{
background-image:linear-gradient(160deg,var(--card-a),var(--card-b));
}
}

@media print{

html,
html[data-theme="dark"]{
--ink:#12304A;
--mute:#57718A;
--line:#D2E2EE;
--accent:#0A7EA4;
--accent-d:#075B7A;
--tile-hi:#FFFFFF;
--tile-lo:#EEF5FA;
--btn-top:#2BAAD1;
--btn-bot:#0A78A0;
}

body::before{
display:none!important;
}

*{
-webkit-print-color-adjust:exact;
print-color-adjust:exact;
}

.card{
-webkit-backdrop-filter:none;
backdrop-filter:none;
}

.big{
filter:none!important;
}

.gauge div{
opacity:.45;
box-shadow:none!important;
transform:none!important;
}

.gauge div.on{
opacity:1;
outline:3px solid #12304A;
outline-offset:1px;
}

.bar i{
box-shadow:none!important;
}

.bar,
.model-box,
.note,
.res-top,
.sum,
.gauge,
.glabels{
break-inside:avoid;
page-break-inside:avoid;
}

h2,
h3{
break-after:avoid;
page-break-after:avoid;
}

}


/* =====================================================
   TAMBAHAN 2: latar diam (tidak ikut scroll) + foto bergaya rumah sakit
   (hanya visual, tidak mengubah alur aplikasi)
===================================================== */

:root{
--photo-tint:linear-gradient(165deg,rgba(214,236,248,.72) 0%,rgba(232,244,251,.78) 55%,rgba(244,250,253,.84) 100%);
}

html[data-theme="dark"]{
--photo-tint:linear-gradient(165deg,rgba(7,20,32,.74) 0%,rgba(10,26,42,.80) 55%,rgba(12,32,52,.84) 100%);
}

html{
background-color:var(--bg-b);
}

/* Lapisan paling belakang: warna, pola salib, dan cahaya (diam) */

body,
html[data-theme="dark"] body{
background:none;
}

body::after{
content:"";
position:fixed;
top:0;
right:0;
bottom:0;
left:0;
z-index:-1;
pointer-events:none;
background:
url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='64' height='64'%3E%3Cpath d='M32 24v16M24 32h16' stroke='%230A7EA4' stroke-opacity='.10' stroke-width='3.5' stroke-linecap='round'/%3E%3C/svg%3E"),
radial-gradient(900px 520px at 6% -8%, rgba(43,170,209,.25), transparent 60%),
radial-gradient(800px 500px at 100% 0%, rgba(59,197,150,.16), transparent 55%);
}

html[data-theme="dark"] body::after{
background:
url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='64' height='64'%3E%3Cpath d='M32 24v16M24 32h16' stroke='%238FD6F0' stroke-opacity='.06' stroke-width='3.5' stroke-linecap='round'/%3E%3C/svg%3E"),
radial-gradient(900px 520px at 6% -8%, rgba(47,159,203,.22), transparent 60%),
radial-gradient(800px 500px at 100% 0%, rgba(59,197,150,.10), transparent 55%);
}

/* Lapisan foto dokter: diam, diberi tirai warna rumah sakit */

body::before,
html[data-theme="dark"] body::before{
z-index:-2;
top:-24px;
right:-24px;
bottom:-24px;
left:-24px;
opacity:1;
filter:blur(2.5px) saturate(.85);
background-size:cover;
background-position:center;
background-repeat:no-repeat;
}

body:has(#v1.on)::before{ background-image:var(--photo-tint),url("/bg/beranda"); }
body:has(#v2.on)::before{ background-image:var(--photo-tint),url("/bg/data-pasien"); }
body:has(#v3.on)::before{ background-image:var(--photo-tint),url("/bg/hasil-sitologi"); }
body:has(#v4.on)::before{ background-image:var(--photo-tint),url("/bg/hasil"); }

@media print{

body::after{
display:none!important;
}

html{
background:#fff!important;
}

}

</style>

</head>

<body>

<div class="wrap">

<div class="brand">
<span class="brand-title">Skrining Awal Tumor Payudara</span>
<button type="button" class="theme-toggle" id="themeToggle" aria-label="Ubah tema tampilan">Mode gelap</button>
</div>

{% if error %}

<div class="err" role="alert">
<strong>Perhatian:</strong> {{ error }}
</div>

{% endif %}


<!-- =================================================
     HALAMAN 1: BERANDA
================================================== -->

<section
class="view {% if active_view == 1 %}on{% endif %}"
id="v1"
>

<div class="card">

<h1>Analisis Hasil Sitologi untuk Mendukung Penilaian Dokter</h1>

<p>
Dokter dapat memasukkan data pasien dan angka dari
lembar hasil sitologi untuk memperoleh keluaran
klasifikasi dari dua model machine learning
yang telah dilatih.
</p>

<div class="guide">

<div>
<h3>1. Isi data pasien</h3>
Nama, usia, kontak, dan gejala pasien.
</div>

<div>
<h3>2. Masukkan hasil sitologi</h3>
30 nilai fitur Mean, SE, dan Worst.
</div>

<div>
<h3>3. Lihat hasil</h3>
Probabilitas dan klasifikasi dari kedua model.
</div>

</div>

<div class="note">

<strong>Batasan penggunaan:</strong>

Sistem ini dikembangkan menggunakan dataset publik Breast Cancer Wisconsin 
(Diagnostic), bukan data pasien nyata. Hasil prediksi merupakan estimasi model 
yang hanya digunakan sebagai informasi pendukung dan 
tidak menggantikan diagnosis maupun keputusan klinis dokter.

</div>

<div class="row">

<button type="button" class="btn" data-go="2">
Mulai skrining
</button>

</div>

</div>

</section>


<!-- =================================================
     FORM INPUT
================================================== -->

<form method="POST" id="screenForm" novalidate>


<!-- HALAMAN 2: DATA PASIEN -->

<section
class="view {% if active_view == 2 %}on{% endif %}"
id="v2"
>

<div class="steps">

<span class="done">1. Data pasien</span>
<span>2. Hasil sitologi</span>
<span>3. Hasil</span>

</div>

<div class="card">

<h2>Data pasien</h2>

<div class="grid">

<div>

<label for="nama">Nama pasien *</label>

<input
type="text"
id="nama"
name="nama"
autocomplete="name"
value="{{ form.get('nama','') }}"
required
aria-describedby="namaError"
>

<div class="hint" id="namaError" role="alert" style="color:#A32121">
{% if error == 'Nama pasien wajib diisi.' %}{{ error }}{% endif %}
</div>

</div>

<div>

<label for="usia">
Usia <span class="opt">(opsional)</span>
</label>

<input
type="number"
id="usia"
name="usia"
min="1"
max="120"
step="1"
value="{{ form.get('usia','') }}"
>

</div>

<div>

<label for="kontak">
Kontak <span class="opt">(opsional)</span>
</label>

<input
type="tel"
id="kontak"
name="kontak"
autocomplete="tel"
value="{{ form.get('kontak','') }}"
>

</div>

</div>

<h3 style="margin-top:1.5rem">

Gejala yang dialami

<span class="opt">(opsional)</span>

</h3>

<div class="note">

Gejala hanya dicatat sebagai informasi dan tidak
digunakan sebagai fitur prediksi oleh kedua model.

</div>

<div class="sym">

{% for symptom in symptoms %}

<label>

<input
type="checkbox"
class="symptom"
name="gejala"
value="{{ symptom }}"
{% if symptom in form.getlist('gejala') %}checked{% endif %}
>

{{ symptom }}

</label>

{% endfor %}

</div>

<div class="row">

<button type="button" class="btn sec" data-go="1">
Kembali
</button>

<button type="button" class="btn" id="toFeatures">
Lanjut
</button>

</div>

</div>

</section>


<!-- =================================================
     HALAMAN 3: INPUT FITUR
================================================== -->

<section
class="view {% if active_view == 3 %}on{% endif %}"
id="v3"
>

<div class="steps">

<span class="done">1. Data pasien</span>
<span class="done">2. Hasil sitologi</span>
<span>3. Hasil</span>

</div>

<div class="card">

<h2>Hasil sitologi (30 nilai)</h2>

<p>
Salin angka dari lembar hasil pemeriksaan.
Gunakan titik untuk desimal, misalnya 0.0978.
Semua nilai harus diisi dan tidak boleh negatif.
</p>

<div class="note">

Rentang dataset ditampilkan sebagai panduan pemeriksaan
input. Nilai di luar rentang akan diberi peringatan,
tetapi tetap dapat diproses setelah diperiksa kembali.

</div>

<div class="tabs" role="tablist">

{% for key, group in feature_groups.items() %}

<button
type="button"
class="tab"
role="tab"
aria-selected="{{ 'true' if loop.first else 'false' }}"
data-tab="{{ loop.index0 }}"
>

{{ group.title }}

<span
class="hint"
style="color:inherit"
id="count{{ loop.index0 }}"
>(0/10)</span>

</button>

{% endfor %}

</div>


{% for key, group in feature_groups.items() %}

{% set group_index = loop.index0 %}

<div
class="pane {% if loop.first %}on{% endif %}"
id="pane{{ group_index }}"
>

<h3>{{ group.title }}</h3>

<p class="hint">
{{ group.description }}
</p>

<div class="grid">

{% for feature in group.features %}

<div>

<label for="{{ feature }}">
{{ feature_labels[feature] }}
</label>

<input
type="number"
step="{{ feature_steps[feature] }}"
min="0"
inputmode="decimal"
id="{{ feature }}"
name="{{ feature }}"
value="{{ form.get(feature,'') }}"
data-group="{{ group_index }}"
data-min="{{ feature_ranges[feature][0] }}"
data-max="{{ feature_ranges[feature][1] }}"
data-step="{{ feature_steps[feature] }}"
>

<div class="hint">

{{ feature_descriptions[feature] }}

<br>

Rentang dataset:
{{ feature_ranges[feature][0] }}–{{ feature_ranges[feature][1] }}.

</div>

</div>

{% endfor %}

</div>

</div>

{% endfor %}

<div id="clientError" class="err" hidden></div>

<div class="row">

<button type="button" class="btn sec" id="sampleBenign">
Isi contoh jinak
</button>

<button type="button" class="btn sec" id="sampleMalignant">
Isi contoh ganas
</button>

<button type="button" class="btn sec" data-go="2">
Kembali
</button>

<button type="submit" class="btn" id="submitBtn">
Lihat hasil
</button>

</div>

</div>

</section>

</form>


<!-- =================================================
     HALAMAN 4: HASIL SKRINING
     DESAIN MENGIKUTI HTML TEMAN
================================================== -->

{% if result %}

<section class="view on" id="v4">

<div class="steps">

<span class="done">1. Data pasien</span>
<span class="done">2. Hasil sitologi</span>
<span class="done">3. Hasil</span>

</div>

<div class="card" id="result">

<h2>Hasil skrining</h2>


<!-- PROBABILITAS DAN KLASIFIKASI -->

<div class="res-top">

<div>

<div class="big">
{{ result.logistic.probability }}%
</div>

<div class="hint">
Probabilitas ganas
</div>

</div>

<div>

<div class="hint">
Klasifikasi model Logistic Regression
</div>

<h3 style="margin:0">
{{ result.logistic.label }}
</h3>

</div>

</div>


<!-- GAUGE LEVEL RISIKO -->

<h3 style="margin-top:1.5rem">
Level risiko
</h3>

<strong>
Level {{ result.logistic.level_number }} dari 5:
{{ result.logistic.level_label }}
</strong>

<div class="gauge">

{% for color in risk_colors %}

<div
class="{% if loop.index == result.logistic.level_number %}on{% endif %}"
style="background:{{ color }}"
></div>

{% endfor %}

</div>

<div class="glabels">

<span>Sangat rendah</span>
<span>Rendah</span>
<span>Sedang</span>
<span>Tinggi</span>
<span>Sangat tinggi</span>

</div>


<!-- PERTIMBANGAN UNTUK DOKTER -->

<h3 style="margin-top:1.25rem">
    Pertimbangan untuk dokter
</h3>

<div class="note" id="r-adv">
    {{ result.logistic.advice }}
</div>


<!-- PERINGATAN DI LUAR RENTANG -->

{% if out_of_range %}

<div
class="note"
style="border-color:#E5A52A;background:var(--warn-bg)"
>

<strong>Perhatian:</strong>

{{ out_of_range|length }} nilai berada di luar rentang
dataset, yaitu:

{{ out_of_range|join(', ') }}.

Periksa kembali angka dan satuannya.
Rentang ini bukan rentang klinis.

</div>

{% endif %}


<!-- PERBANDINGAN DUA MODEL -->

<h3 style="margin-top:1.5rem">
Perbandingan hasil kedua model
</h3>

<div class="model-comparison">

<div class="model-box">

<strong>Logistic Regression</strong>

<div class="pct">
{{ result.logistic.probability }}%
</div>

<strong>
{{ result.logistic.classification }}
</strong>

<div class="hint">
{{ result.logistic.label }}
</div>

</div>


<div class="model-box">

<strong>Decision Tree</strong>

<div class="pct">
{{ result.tree.probability }}%
</div>

<strong>
{{ result.tree.classification }}
</strong>

<div class="hint">
{{ result.tree.label }}
</div>

</div>

</div>


<div class="note">

<strong>Kesepakatan klasifikasi:</strong>

{% if result.agreement %}

Kedua model menghasilkan klasifikasi yang sama.

{% else %}

Kedua model menghasilkan klasifikasi berbeda.
Hasil perlu ditinjau secara cermat oleh dokter.

{% endif %}

</div>


<!-- FITUR PALING BERPENGARUH -->

<h3 style="margin-top:1.5rem">
Fitur yang paling berpengaruh
</h3>

{% if contributions %}

{% set max_contribution = contributions[0].abs_value %}

{% for item in contributions %}

<div class="bar">

<span title="{{ item.label }}">
{{ item.label }}
</span>

<i
style="
width:{{ (item.abs_value / max_contribution * 100)|round(1) }}%;
{% if item.value < 0 %}background:#7A9BA0;{% endif %}
"
></i>

<span>
{{ item.direction }}
</span>

</div>

{% endfor %}

<div class="hint">

Naik: kontribusi positif terhadap skor model.
Turun: kontribusi negatif terhadap skor model.

Nilai kontribusi dihitung dari koefisien Logistic Regression.
Interpretasinya bergantung pada preprocessing model.

</div>

{% else %}

<div class="note">

Model Logistic Regression tidak menyediakan koefisien
fitur yang dapat digunakan untuk menampilkan kontribusi
secara langsung. Prediksi tetap menggunakan model asli.

</div>

{% endif %}


<!-- RINGKASAN PASIEN -->

<h3 style="margin-top:1.5rem">
Ringkasan data Pasien
</h3>

<dl class="sum">

<dt>Nama pasien</dt>
<dd>{{ patient.nama }}</dd>

<dt>Usia</dt>
<dd>{{ patient.usia or '-' }}</dd>

<dt>Kontak</dt>
<dd>{{ patient.kontak or '-' }}</dd>

<dt>Catatan gejala</dt>

<dd>
{{ patient.gejala or 'Tidak ada gejala yang dicatat' }}
</dd>

</dl>


<!-- CATATAN MEDIS -->

<div class="note">

<strong>Penting:</strong>

Probabilitas yang ditampilkan merupakan hasil estimasi model, 
bukan diagnosis medis atau kepastian adanya keganasan.  

Hasil ini digunakan sebagai informasi pendukung bagi dokter dan perlu diinterpretasikan 
bersama kondisi klinis pasien serta hasil pemeriksaan penunjang.

Keputusan diagnosis dan tindak lanjut tetap 
didasarkan pada penilaian medis secara menyeluruh.

</div>


<!-- TOMBOL CETAK DAN ULANG -->

<div class="row">

<button
type="button"
class="btn"
onclick="window.print()"
>
Cetak hasil
</button>

<a class="btn sec" href="/">
Skrining ulang
</a>

</div>

</div>

</section>

{% endif %}


<footer>

Skrining Awal Tumor Payudara<br>

Sumber data model: Wolberg et al. (1995), Breast Cancer Wisconsin (Diagnostic)<br>

UCI Machine Learning Repository, CC BY 4.0.

</footer>

</div>


<!-- =================================================
     JAVASCRIPT
================================================== -->

<script>

// =====================================================
// NAVIGASI HALAMAN
// =====================================================

function go(n){

    document.querySelectorAll('.view').forEach(v => {

        v.classList.toggle(
            'on',
            v.id === 'v' + n
        );

    });

    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });

}


document.querySelectorAll('[data-go]').forEach(b => {

    b.addEventListener('click', () => {

        go(Number(b.dataset.go));

    });

});


// =====================================================
// NAVIGASI TAB MEAN, SE, DAN WORST
// =====================================================

function changeTab(i){

    document.querySelectorAll('.tab').forEach((t, j) => {

        t.setAttribute(
            'aria-selected',
            j === i ? 'true' : 'false'
        );

    });

    document.querySelectorAll('.pane').forEach((p, j) => {

        p.classList.toggle('on', j === i);

    });

}


document.querySelectorAll('.tab').forEach(b => {

    b.addEventListener('click', () => {

        changeTab(Number(b.dataset.tab));

    });

});


// =====================================================
// HITUNG JUMLAH INPUT PADA SETIAP TAB
// =====================================================

function updateCounts(){

    for(let i = 0; i < 3; i++){

        const fields = document.querySelectorAll(
            '#pane' + i + ' input[type=number]'
        );

        const count = [...fields].filter(
            x => x.value.trim() !== ''
        ).length;

        const target = document.getElementById(
            'count' + i
        );

        if(target){

            target.textContent = `(${count}/10)`;

        }

    }

}


document.querySelectorAll(
    '#screenForm input[type=number][data-group]'
).forEach(x => {

    x.addEventListener('input', updateCounts);

});

updateCounts();


// Contoh nilai dari data WDBC (30 fitur sesuai urutan model).
// Nilai ini hanya untuk demonstrasi pengisian, bukan data pasien.
const exampleValues = {
    benign: [
        13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,
        0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,
        15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259
    ],
    malignant: [
        17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,
        1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,
        25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189
    ]
};

function fillExample(kind){
    document.querySelectorAll('#screenForm input[type="number"][data-group]')
        .forEach((input, index) => {
            input.value = exampleValues[kind][index];
            input.dispatchEvent(new Event('input', {bubbles:true}));
            input.dispatchEvent(new Event('change', {bubbles:true}));
        });
    updateCounts();
    changeTab(0);
}

document.getElementById('sampleBenign')?.addEventListener('click', () => fillExample('benign'));
document.getElementById('sampleMalignant')?.addEventListener('click', () => fillExample('malignant'));


// =====================================================
// PILIHAN GEJALA
// =====================================================

const symptoms = [
    ...document.querySelectorAll('.symptom')
];


symptoms.forEach(x => {

    x.addEventListener('change', () => {

        if(
            x.value === 'Tidak ada gejala' &&
            x.checked
        ){

            symptoms.forEach(y => {

                if(y !== x){
                    y.checked = false;
                }

            });

        }

        else if(x.checked){

            symptoms.forEach(y => {

                if(y.value === 'Tidak ada gejala'){
                    y.checked = false;
                }

            });

        }

    });

});


// =====================================================
// VALIDASI DATA PASIEN SEBELUM LANJUT
// =====================================================

document.getElementById('nama')?.addEventListener('input', () => {
    if(document.getElementById('nama').value.trim()){
        document.getElementById('namaError').textContent = '';
    }
});


document.getElementById('toFeatures')?.addEventListener(
'click',
() => {

    const nama = document.getElementById('nama');

    const namaError = document.getElementById('namaError');
    if(!nama.value.trim()){
        namaError.textContent = 'Nama pasien wajib diisi.';
        go(2);
        nama.focus();
        return;
    }
    namaError.textContent = '';

    const usia = document.getElementById('usia');

    if(
        usia.value &&
        (
            !Number.isInteger(Number(usia.value)) ||
            Number(usia.value) < 1 ||
            Number(usia.value) > 120
        )
    ){

        alert(
            'Usia harus berupa bilangan bulat antara 1 sampai 120.'
        );

        usia.focus();

        return;

    }

    go(3);

});


// =====================================================
// VALIDASI SEBELUM PREDIKSI DIKIRIM KE FLASK
// =====================================================

document.getElementById('screenForm')?.addEventListener(
'submit',
function(e){

    const nama = document.getElementById('nama');

    const namaError = document.getElementById('namaError');
    if(!nama.value.trim()){
        e.preventDefault();
        namaError.textContent = 'Nama pasien wajib diisi.';
        go(2);
        nama.focus();
        return;
    }
    namaError.textContent = '';

    const usia = document.getElementById('usia');

    if(
        usia.value &&
        (
            !Number.isInteger(Number(usia.value)) ||
            Number(usia.value) < 1 ||
            Number(usia.value) > 120
        )
    ){

        e.preventDefault();

        alert(
            'Usia harus berupa bilangan bulat antara 1 sampai 120.'
        );

        go(2);

        usia.focus();

        return;

    }

    const fields = [
        ...document.querySelectorAll(
            '#screenForm input[type=number][data-group]'
        )
    ];

    let first = null;

    fields.forEach(x => {

        x.classList.remove('bad');

        x.classList.remove('out');

        const value = x.value.trim();

        const number = Number(value);

        if(
            value === '' ||
            !Number.isFinite(number) ||
            number < 0
        ){

            x.classList.add('bad');

            if(!first){
                first = x;
            }

            return;

        }

        const min = Number(x.dataset.min);

        const max = Number(x.dataset.max);

        if(number < min || number > max){

            x.classList.add('out');

        }

    });


    if(first){

        e.preventDefault();

        const err = document.getElementById(
            'clientError'
        );

        err.hidden = false;

        err.textContent =
            'Semua 30 fitur harus diisi dengan angka valid dan tidak negatif.';

        go(3);

        changeTab(Number(first.dataset.group));

        first.focus();

        return;

    }


    const button = document.getElementById(
        'submitBtn'
    );

    button.disabled = true;

    button.textContent = 'Memproses prediksi...';

});


// =====================================================
// MENAMPILKAN HALAMAN SESUAI KONDISI FLASK
// =====================================================

{% if error %}

go({{ active_view }});

{% elif result %}

go(4);

{% endif %}

</script>

<script>
(function(){
  var b=document.getElementById('themeToggle');
  var r=document.documentElement;
  if(!b){ return; }
  function label(){
    b.textContent = r.getAttribute('data-theme')==='dark' ? 'Mode terang' : 'Mode gelap';
  }
  label();
  b.addEventListener('click',function(){
    var n = r.getAttribute('data-theme')==='dark' ? 'light' : 'dark';
    r.setAttribute('data-theme',n);
    try{ localStorage.setItem('tema',n); }catch(e){}
    label();
  });
})();
</script>

</body>
</html>
"""


# =========================================================
# ROUTE FLASK
# =========================================================

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    error = None
    patient = {}
    out_of_range = []
    contributions = []

    form = request.form

    active_view = 1

    if request.method == "POST":

        active_view = 3

        try:

            # ---------------------------------------------
            # DATA PASIEN
            # ---------------------------------------------

            nama = request.form.get(
                "nama", ""
            ).strip()

            usia = request.form.get(
                "usia", ""
            ).strip()

            kontak = request.form.get(
                "kontak", ""
            ).strip()

            if not nama:

                active_view = 2

                raise ValueError(
                    "Nama pasien wajib diisi."
                )

            if usia:

                try:

                    usia_int = int(usia)

                except ValueError:

                    active_view = 2

                    raise ValueError(
                        "Usia harus berupa angka bulat."
                    )

                if not 1 <= usia_int <= 120:

                    active_view = 2

                    raise ValueError(
                        "Usia harus berada antara 1 sampai 120 tahun."
                    )

            selected = [
                s for s in request.form.getlist("gejala")
                if s in SYMPTOMS
            ]

            if "Tidak ada gejala" in selected:

                selected = ["Tidak ada gejala"]

            patient = {
                "nama": nama,
                "usia": usia,
                "kontak": kontak,
                "gejala": ", ".join(selected),
            }

            # ---------------------------------------------
            # MENGAMBIL 30 NILAI FITUR
            # ---------------------------------------------

            values = {}

            for feature in FEATURES:

                raw = request.form.get(
                    feature, ""
                ).strip()

                if not raw:

                    raise ValueError(
                        f"Nilai {get_feature_label(feature)} belum diisi."
                    )

                try:

                    value = float(raw)

                except ValueError:

                    raise ValueError(
                        f"Nilai {get_feature_label(feature)} harus berupa angka."
                    )

                if not math.isfinite(value) or value < 0:

                    raise ValueError(
                        f"Nilai {get_feature_label(feature)} "
                        "harus berupa angka valid dan tidak negatif."
                    )

                values[feature] = value

            # ---------------------------------------------
            # MEMERIKSA RENTANG DATASET
            # ---------------------------------------------

            for feature, value in values.items():

                prefix, name = feature.split("_", 1)

                low, high = RANGES[prefix][name]

                if value < low or value > high:

                    out_of_range.append(
                        get_feature_label(feature)
                    )

            # ---------------------------------------------
            # MEMBUAT DATAFRAME
            # ---------------------------------------------

            input_data = pd.DataFrame(
                [[values[f] for f in FEATURES]],
                columns=FEATURES,
            )

            # ---------------------------------------------
            # PREDIKSI LOGISTIC REGRESSION
            # ---------------------------------------------

            logistic_result = predict_model(
                logistic_model,
                input_data,
            )

            # ---------------------------------------------
            # PREDIKSI DECISION TREE
            # ---------------------------------------------

            tree_result = predict_model(
                decision_tree_model,
                input_data,
            )

            # ---------------------------------------------
            # FITUR YANG BERPENGARUH
            # ---------------------------------------------

            contributions = get_feature_contributions(
                logistic_model,
                input_data,
            )

            # ---------------------------------------------
            # HASIL AKHIR
            # ---------------------------------------------

            result = {
                "logistic": logistic_result,
                "tree": tree_result,

                "agreement": (
                    logistic_result["classification"]
                    == tree_result["classification"]
                ),
            }

            active_view = 4

        except ValueError as exc:

            error = str(exc)

        except Exception:

            app.logger.exception(
                "Terjadi kesalahan ketika menjalankan prediksi."
            )

            error = (
                "Terjadi kesalahan ketika model memproses data. "
                "Periksa file model, dependensi, dan kesesuaian fitur training."
            )

    # =====================================================
    # MENYIAPKAN DATA FITUR UNTUK HTML
    # =====================================================

    feature_ranges = {}

    for feature in FEATURES:

        prefix, name = feature.split("_", 1)

        feature_ranges[feature] = RANGES[prefix][name]

    feature_steps = {
        feature: get_feature_step(feature)
        for feature in FEATURES
    }

    # =====================================================
    # MENAMPILKAN HALAMAN
    # =====================================================

    return render_template_string(

        HTML,

        result=result,

        error=error,

        patient=patient,

        form=form,

        symptoms=SYMPTOMS,

        feature_groups=GROUPS,

        feature_labels={
            f: get_feature_label(f)
            for f in FEATURES
        },

        feature_descriptions={
            f: FEATURE_DESCRIPTIONS[f.split("_", 1)[1]]
            for f in FEATURES
        },

        feature_ranges=feature_ranges,

        feature_steps=feature_steps,

        active_view=active_view,

        out_of_range=out_of_range,

        contributions=contributions,

        risk_colors=RISK_COLORS,

    )



# =========================================================
# GAMBAR LATAR HALAMAN (tertanam, hanya tampilan)
# =========================================================

BACKGROUND_IMAGES = {
    "beranda": base64.b64decode(
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwg"
        "IyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgo"
        "KCgoKCgoKCgoKCgoKCj/wgARCAFvAmQDASIAAhEBAxEB/8QAGwAAAQUBAQAAAAAAAAAAAAAAAgABAwQFBgf/xAAYAQEBAQEB"
        "AAAAAAAAAAAAAAAAAQIDBP/aAAwDAQACEAMQAAAB9HMD1h5opZpAYQAkNjJJETEPVtVpqGQD1mzFLHKBMVgmBFaMwhklK0Fi"
        "uTEJqnZwkzspJK5CQTpKkkJJxk6EkhOkEJCQpIFJQkkMzqmSQkkOkkUschJUt1AIzjgCGMOnIbNdWEnQGJ70pYpZpBIERiYW"
        "MkkchJXrWa0sJC9zZZIEhIYgOqkckcqSUrQWK6TmJKhy/MDu8LiA1O23vLLB7xP4l6RL0ySh3ZwU7DpkOmQ6SDZ2IEkCiGEk"
        "hk7UzpDOzidnRSRyEtO5UIoZoJBqWKrNg5YhJJegIS3pSxyyoDCAAxsFJIRCSvWsV5a5CdzOkhiEgCYimBxzToUDUlo7zdoF"
        "wduTl2AiNHZlrXXu51NldBzSev73ifslzYdlK4oQnFx0ziSQaTldJCEhhOkMztSSQkkJ2dFJHIs1O5USGCeCSKvZhknEFYai"
        "S9O7FvSljklQmERiQ2C6SOTOr17FeWtIB3M6SGdnGdnKUUkE0pK8sW0by4HmXV8d1xU0JfTefXmtTSHzequ1ozI4b0+hrHlH"
        "fczR9Pl9xaNRIIOkhxnBOJWpJwiZyskhCQwnSGZ2pJISSE7OiMDLFW0y0YtNJlrUUYzbTWYq2kTu60UkZzRATREztYKdI7s6"
        "vBPEtSSQkSJwHJyNSIo19VS5Euk5Ug0uZs4Tn9rnek9K6nI2vJ6qEc8fPo0gOSR1YLKfm3sPkXXl6l1fivtfXgkmlJ40SPCi"
        "V4UTMJFZOwKShJnEztSSQkkOkmUYGtirapqDRhEzQImaNEijR0qZqdkgU6sjZ2BTqx0lK8M0BXauEW1TEuqi6XmqotPVKrDw"
        "ol5HquVrk8Dex9a9Uzr3N+X0T7XKauenRZ2liZuZo5WlvHR+b+hYMvm/u/gPtvp8m2iaZZiYSdDO7hSxyrWZwVJNDpnpM7CS"
        "QkkjpkjkBlmrbqNQBIEREnGToZOjoAMdZd2dUxDKAmFjJ2sdJBQT15c4U8jImAd0EiQxIqRJxuO7LiNTlc+7nt+x0dqh4/Xl"
        "2ZZs6kr3ILKU8c81NhbeJZ5B7j4h7l7PDtsbZgomGd0JOgpI5FrCQKmQwTiVMkhJMOmQ6ZMucZraq26qwgYQCdDIhEkjfJn0"
        "ZJQ7JACYWMzqxJIeCeuUWJoZiEF0okSVIwMJ2cXC91w+5yWTpYrXtTYWx4fbVv4NtrYiybV0cNC7manmnpHjPbzQex+Les9/"
        "P271bWamdhOzidnUpI5EqgcasyaCICpVLeLcy1Iub746huWWs93f5jpfN1eSI5q7WtViCOWOI3ZBCQiSR0CdqdJxmdgRIbBS"
        "VidnhV7EFUWdoTOwKSiRJUiEg2SH4TV5/pOaytHPl0PUvFO08/o7Gvfk5d469a1rTWzy8557gJIvV4i7zhdxn1Db5LqMJxFt"
        "JXjdZUBhyRSrViliBEgCKMxYG9y+sW+U6nivTzvPWm1nZ67hO48vYzikxvRrWa5DFLFEaZDpISZG+7PrKIXVJnlESGwU7WJ2"
        "eFBYgqgnGHFxGSaWRMkr0umr6c9y3bcDXJ7nNaOo1aaKWW8Vrj36XpeS0vN6dg6J1YwtSFnzTG9q5n2+Ljuw6PdSlpxlkhsQ"
        "SsUZJKURrLPXsFWGeFQjkjhSxSEfL9Jy3TGrw/Y8p6OZT2FVfvPPe+8/SzJHJz3pV7FchhmgiNkwbDETqBL06J9ZF3UoulTC"
        "QjMQonZ0UE8FtESHMQuAmYNJL+Zt9eYQnBy7UfK/W86zxkut47WZRlry9tN1fPcPRJarW/N6bEwGizb1Gzp+D9E8w9vh9FnZ"
        "lRDISsEkVJ2tyVUYylarWarQTwkcUsMKSGQio3pwec7fBKU9i7Lx/Q1bBbkiks1K9ivUNeevEQiKylFcGRo2nZ9ZdJ5oU7Iw"
        "uNJkyOmSFBNCUBlCUIp4yEiMmu15qCFyoK0sRxfAeg8DqFqZHoC9ziS7fPfB2q9vx+4iilGqvf3z6Pz/ANE471+LX0Oc6MeW"
        "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"
    ),
    "data-pasien": base64.b64decode(
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwg"
        "IyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgo"
        "KCgoKCgoKCgoKCgoKCj/wgARCAGYAmQDASIAAhEBAxEB/8QAGwAAAQUBAQAAAAAAAAAAAAAAAgABAwQFBgf/xAAZAQEBAQEB"
        "AQAAAAAAAAAAAAAAAQIDBAX/2gAMAwEAAhADEAAAAfSzYgkKCTOO4uO7IdJGTgdDg9OU9+lf1DmhlmkBileKaHWQTtYPB95w"
        "bOCJDNgxDDVrNddXruQ7Dl3COYdYiaQZYxkEoxzR46tZrWNYQurkTYjLwd/BFNGdz7YOJq8fTPKwRxNbc57Havfo6hHLZcjo"
        "3+P3ye7R2+3mpTxTZ30Cdiq6dWdOiljlDTo7xwJCcSHdnHdISSE7OZeD0GDvnLfpXtZkljkVCQkEM0OsszsDwvd8Ncc4JNNg"
        "JNDVrNddHsuL7Tl3GOSrcytDIOJCVI5os9AsV57lMmuSKCaM/D67UrgLvquumZILZ2sfdxuPpyc58vPTUsVZdYuucJU8+6nk"
        "+nHpejw+m7+bJg1OVzrtmibGxdOM6QUkcpIkjtzE0RIh3ZBIXHTIdxcz8Hewd85b1C/rMssUrSEhSCKeC5FJUPD9xxVzzIkM"
        "0IkMNBPCtzteH7bl3IDC5ATEETYqx2Ic9AmC3c1+l2b7Ofeo6dVFLKRkdIpYe7yE13efn3uPfn47EU6Q25DuYsnUypOUvFpe"
        "jywdRz27rNPi/QOYljtc3smlpZk8tiWmct2ShZLagUvoJA6G4IkcHCcXHTId2coYPQc9vnJeoX9ZlmhmmkzjUcM0NyIkNjcX"
        "2nHXPKM7TQCQw0M0Usvc8P3PPu9ezDcwo4xyjIGOSPO5O/z+huQqWoEzJ6ds1YjZZM3RgSpxXW83pzvceb7vn9Xe5uRoJVci"
        "UY7o9OWVymrW9Hk39COTl2q42nnnA9hx/VbzZs07uNOmnJIDz7LK0knoJRljcjgQRC4Ts4knE7OU+e6Hnd4K9QvaxNYr2Jpw"
        "IajhmhuRZ2sHkOv5O549naaESGBiljll7ng+859zjkG5iCSOGZ2Av0e/mtASiuRGAaprUw7J9Hzr0PpxtnIHH0eV6GB2Xq8f"
        "BaER+D6l6aNTV+3l7esQZlrE9vzszpbh8e7NZpZ1UoX804zbydPedW/FaxppbucZutV3zNVpXPcExY25JxyZx0kJJglGZV5z"
        "o+b3h7+ff1iexWsNEJARxSw3Is7WNyvVczc8SztNCJjAxSxSrvOC7Xl6bw0Qzq8FIauDUmZ6PsKV3XMBcjHOOOtvD2YU849E"
        "8/7jvw2EJ+f0+b3N3A9Hlyc9j8X0rYxjnVzb5/e7+fku3l3Ovnq8/t4nPrQhrS2S1bddOP0IbPTPU2Jtbl0z+e6EkkqXqFVV"
        "1iHeUs2B7JFVXGKr59ypEDIclYAsqzUuSv51vWb1inaWQXAGKSOwGdrG5rpecueGZ2zoWdgY5YpY+34juuPqYSix2eNgD6Lm"
        "e41z17VK704RyRSxl1dXIrRuY+keddbyPQ9+G7qc50HDvV4H0fldY4uvcp56T3c7ex1ubPO916PGzsXHvz+To42s1bhxletK"
        "k52yzdcdd1HI9Ry6VZJa0NRuTW66oKJhqEWVA8TMBFHTqsbbWFFdWHrJw+p5bWRsVbWsWbVCxVsY2JIkKOhVPz2/g3PBs7Z0"
        "wkyiJjEXW8qsdeqHlnm+mXMudJ6L417LcwbWJZS/IDQWZpVqyL/Och34dD0PAS75+l2eX6TzerRyNSPOvN8rsOY68cvr+K77"
        "HTorMoyyY0FIirXBsr5skFjG12XCztfI7cey38rQ49iN5VpadDWjCQK5mkgOamKMoMgckpXIV6E8vThyByHleq5TeI7+ff1i"
        "SeKzQxWoEiZR2SOCsPD2cg4Bks6ZkpWZ0CnMjUggk5Det+T+tTV6pFazsrNSYlrWOe1jjO7ivdMWK7jy6+aek4GF6fL6hbqW"
        "fJ68bi/ROH1nC9P43srksEBx1ZrkNkWXYzLE9nbMTo5rUvnGTdg9Xk7nQo6Hm9T2qGlLGcuOldUFqXzhPOpjgMmOAyZweWPo"
        "+U6QmdFJX5fqeX1ILlO7rEl6ndskinjWlFarawLs9j5Orl2efJ2xtmdpWTsIwIdk4pF2i5fqkFzG8Ctp4S7pVLJY4buvOu3D"
        "t5kuXaKpoUJcvg/QfOPT5vWNbF2/N6Zef36MvK61SkW7sekDgvk6yOzb6EryRBLNlaXPWcJNW1/T5+wneHzeia5j6ksef02J"
        "ZkpKyV4izqU4CJzhIsSV5IaxBUXrCoXEHmOj5yyC3TsaxZ0MvSqyBgQVrdXWYk43KzNPPs88Z2xtmSlZJCISHIZTovUMLZx0"
        "lKApauTs59Y3TYcJ0HC71jpz0StQ89qrcqLj+e+pcz059NsZmnz6Sijl5fG7XlpdrndKfWc+5oI0uc0+aTQ2MGhZp5uXNXM9"
        "5w3f9uM9aXP4d7e1znSS6VKSycorisy3jRIULrYkqyxPJXMs1J2lt7XN7qFDJAcVuxtvncv5WpVkDEiq2q9zAnbWRoX6Kedi"
        "Q52KSlZOwiEgui570GXq1PTz0uFDJCgMjKzuhpVx29Dds0refpSw1rVcq8Z2vI2dno0b0pIClfmOn56KmkSqwT55qHVnMHne"
        "lzNZtcu9azB9P5jrt5gztHJ57j6TktSOrKrKukhUcOLBrJqFlsyVJC3JWkltHXklr6ubOXxUkudFaq6xLq5OpvFwSEirWatz"
        "GyWsjTuVDzkJIs6TJSskhGLkvr3kXs2dWxF5um8HFWeiyZUsunTpZ5Tv8rQT0TZw90ghmjlzeW6bmrO1u0b0rSxmrc90PJru"
        "sJoOJNWN+ObJKFUbZw82lr3PawTgvNY2zjXNCxAZ1OpwPRr2CFZvBQjHvJNEixLVkluS1JZbUlaUlSGXWeGSIK09bWZNTK0t"
        "4vikR1bVa5gZx1lVrNavOYpYsaTO0qSQiElk9W8q9pzblds7O5PMfQuaa2pfPfV7nnz0M4yuG9F46zsu08j9YGElm4nPdFl1"
        "0l2nagk0Sz8X2fES7lzL0Uw5qtxrcoWqFzBHDOLZyzs186GMq5l+nrNGOfFKVzZ9FlkVhZvlJRS7yJSvEZE4ViO0qleeVjMo"
        "WtBpZ3z1e9izV+/gRaz2TcpSXrR5HrtYiYx7eYa1qtXnMM0GNJOpWTpU6RN7X4p7TnUQLO59rXM7HH8u/D9ssXpy7na8/wBX"
        "HS/wXW8b14WvVfHvSNc+wTtLmVdCpWhLVoZ3tBnnGjxXXcoXdTJ1rMK/XtrYozU0jsQ3rI3kkK1fRVnOQdk9nnWF7DWLLo5S"
        "Ro8+tXLZnrVYyH0kVbE8hBLIcoKxPKUle5nfA5nV8tz71M7eIwLe2S7mtwk/XzdeOOfTlrQRRaxy1XqJ864qL1GyeQx+vxHk"
        "j+qV7PNvTqajq82/m8vRDy/c8hz7Uq8mrN41q1S1inLg9Z183Iei421rOiXP18dOsg5csdOjzmLO8yzGUs80NixTVorL6aJH"
        "i0MlbHScN3WubEB9OUbDHYVXZgIJSEsFURZVRHMXL1QkKhEab40R0Bc3Mb9gGxubBq0+Pq2+l5fq988Lj+346XMQsFJWkSxF"
        "JKuDb6K0uM3aWO3m5LW2HualglEstVLcem5dloTlrK0cvO5OG64Ofbj1RlWbF2K2+PSw085bFzBbO9cs2bOpq9i+sO1nEWGp"
        "SJdCGtZdjp5x0VXGR0NLOOW92nCddvlYly5N8rsEgVJKDWTvSnJThjLSz0WpjkljKRyF5WgHPMm6fOhHx9SsxrHTV7Hz/vev"
        "nz8joM245TO2MrPSApGsllELNw+KhX0XS5vs+nDPbQWsZq0Glz3uopPbqJNZ5W9vnuBVoNZWOdfUpZN/Q1nL6zb2OPXznM9T"
        "JfKNnvFnfnZ+hLOvP4vR31ny8PVFc+Qw+yCePF6/XPJYOgxu3CK5nafLt1TVbON1+owjNxYwXO8ODFZ0zc6jfDDazcHCY21h"
        "o7MuYscPTvQZOeu3Ng6Rtc+WPK0wlju1XQ00yeyy8HfHcwsuEmjFFiSObWTjO6Y/R9Hy9z1s+df6cXTMjjT5vpjrcThKPo8v"
        "QYIP6PNa7Xiut5ddnK0JfN7ee0OnmzrJ0ybNYmUroSEnQySGdCEzIJLJstcHao9uFTNn0s6j6ua/x7jPIcRGaoEaBaRgUaAR"
        "RIUFRqiV9HGSVG8vutV2IO7T1bKT9Tc1jktbUqWDiVcaWWqo5ZCjKwyGUmljlp8zU5c1hg9BlvXD5D0+LR5fIrevxSRxxdeE"
        "7V2tt6WH13DvR0Yo/N6t9c8s77WryckvQWuZKt2Tki1jrZOQ2dZ3LHB6e+fWlyTY32C46vZ3R8TPnWnkSYu8nkDpY0/Wq3z6"
        "lbaSV0kJ2cSZDpoiaHOKxiuEMxORo0ecy9JqcfTy2vunrFa9Vq756VLLp2Xc+CAhy9ijz61GdY2jAqkMJw5RIjy67TXVdvhb"
        "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"
        "1ms1mFkFkFcLhcLhWCsp2jB2SZkgChdDqUezV/h2vVF90fi47tKgyf0eq/3Jhsh4qm4vqvdq0RQ8LVIca3S5MJtVbati46XQ"
        "KqJmwxybtVNS0+2xkKf8534ps2awBUrLrUW2NGLwsZwyPmf2tcC98cYpoHcARues2rJqu1e1cLhWCsFYLELFWU7LscSoy4oB"
        "yGSHU9urfX2u8Un3f63KyKyKyKzKju98LBDGehU3zjUZ4qGZMcL65CUz4qvhbI6uqG0cs0z5Zwrq6jBcXRltPpsAa3hoqJiY"
        "ypTm9jUUStU5Wm8iDlNj91ebLT4MWH3um5NLGGQbS21tlYOWL1Z6/IvyK8qylWci3X2kleQ5xTHFNyQyQ6lHs1b6+1ypPvHx"
        "VlZWVlpEPvYciUV/rM1A8RHg8tjP/wB29lTS3cp23ZqsZ9TM27Gm/Snh3FHWQU4pmmqAUg/HP5ndZjGoCweU5al8NM+mjKYq"
        "qDckmddO9rGDJ5OK2ZFtOW2VgrNXsVZUuicC8i71d6JeiXJxKc9NemvcmuKb1KPZq31drlTfe343RciVdXVONqhpzdHrMFIL"
        "KF3tY7imbnrFM66Axc03a7xWxbsYuC8YPRY7apKTendZrWcqc4wuUvuexqkPQhaq38Gnn/GoPmDw5100XdLyaRl5aiS0m4Vm"
        "VmVkr9JxcUN3w7ZW2VtFGHiSAgOcE14QemvTXK/fqv1dpUPErZm4mVq3Grcas2ouanD/AB4HYP6v8SBM4cHe2hIGvTR7U9rt"
        "gdbpUtVay0s4vGx1w53FFDswnkjhVM2ZkPDGKQ2BPQLVx+Kh+jTBc/0PMYTvNI3GKQ3f3OHFE/Ca5VyrlXKlLsHuamuagUCm"
        "9Lq6v2an9XfyrlXKuVdyBKpDuUD2qCVBP4XkPC1CsZSrfrKw/wAbXNIrqymWk6g2dPFlG7Jsjbitiu2yAwqdNi36jyXlrGzz"
        "l6uvKe7EPd1gZk/WRzQ/RpQ/Cz4xpgTl8ad7fd3v9r2SZNyWSupHHF+KaWprldN6FX7dS+r9uivz054snNUcifyyJVcmzBRU"
        "ztQqYmtjQU8UczKyndQ1OnVQq6ZnBVU2z52YySMzfSQ7FPLM2ISSF5sivAld0ATYyVSQ4jVjeSi+mg4pRwovkFGzJ9Q9Hyh3"
        "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"
    ),
    "hasil-sitologi": base64.b64decode(
        "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"
        "+P2+j2ea1+L3+iuz62fMdeVk6czqeNtrqczp8X0+LodTHs+n8rpel8h2/L26zH5vSs2jPahIsTZTVohZVVqqSqyUljNOGmK2"
        "Rh2QY2BGNlQIIrhbQSVnGjsz8V7aTQBuABV5f1GP0eTjey4XVZ2xzvz+6zn7efWjTyJJ2Vktly+Z9T5T2+TBfYejjZKizNnO"
        "qc1MlVLKE2U49tNec4vs+Pz7Xev+TfTfm/TfmMtPHvsyz2EZdHkfT+V2tPP1erx6dWaeXsLeV0vB7TO4Z04WQq5uBCVcxRnE"
        "i0wjKISTJRlCG0ybhMKrKYbAVOiNlfnO7x+WsftfPehRgdcgBBDtqp1rfLLfCq51FItphlc65VTx1PIes836eGXRhu9fn1VQ"
        "ljTtpks7c1kWzhLOo1XI5+LsY9vM3dDzHn9fo4cCz5f0tVOedztq4/Y+n87s7+Vr9Pk6dmHTh1uz5v1Hl71WVz4dxKS3020h"
        "IYVzrEASrnWknGSyhOuJuDJSiDrtoijl7+b4/VT1OF150r5mrn68/Z0Y8ue3q7eT0O/nuKDebgOmWwBBZXQHThz9Iejw3wDh"
        "74+eDrnmWh7vJbaEVagx0AEnoDGqbAKqQrDxAuvJdYPk/Tp5wWdfoh9Pwb7Q6ctesMzT6APPvTIPJ7SYE6wGAFYEQBwAJgSr"
        "CHIAYEsIc9V0h5uy54TrLmBrzeszA6WbAuaAOmf/xAAtEAACAgEDAwMEAgMBAQEAAAABAgADEQQQEiAhMQUTMhQiMEEzQAYj"
        "QjQkFf/aAAgBAQABBQJon9WwQbGD8jQ+c7IcRnnIs65nHMUY/vGJ/VO5g6DB1mYnGASzsGYzSrsXxEOf7xidY6jMwGDpMOx6"
        "TB1mEzMzLjAsoGFMdJV/eMTrHU8zF2G+rvXT06j1jUs9vqWqsFOu1NLaD1anUjoMHWY57iCOIole36X8DOBG1QEt1pCLr3I0"
        "uuS/+kYnWOnMuMQQdGouWinX+oW6yM/clpXVa8ro4yv1hlXT+r0uyMrrDB1mOO8EIgEY8ZU3LYdT2AS/Ud9VqGWaR7LAbSAv"
        "Zm5JbpNYHWtw/wDQMTrHRYcAX5I+4iDbMzPXSfonDTjiU0QuBOxhriUM00rW6d9HqxeDB1mETHQwzE+057J0E4j5McZjj7nU"
        "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"
    ),
    "hasil": base64.b64decode(
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwg"
        "IyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgo"
        "KCgoKCgoKCgoKCgoKCj/wgARCAGYAmQDASIAAhEBAxEB/8QAGwAAAQUBAQAAAAAAAAAAAAAAAAECAwQFBgf/xAAZAQEBAQEB"
        "AQAAAAAAAAAAAAAAAQIDBAX/2gAMAwEAAhADEAAAAeoekm4vPb/I9Oe1ocpY1nrZOe0M60ivLDwIFAFEVRqEhAkTJAK9iNlc"
        "jI5qdtaMsxROgJbBUkuSy1c7b5w5vPjr51ednWZdXd5nQ3ntbUB142yOTNUQFQFdG4Ss+SvSiuJH1JokAahhuQ3NadIrLq1Z"
        "86egKyKw1mFyvWkyTJ1mOBzd83ihsyVHcfRY5Tfl3nh7G5g9eNyfIkrds4VjOt+zz0svRLhW83QYx2dKNaspBHFplSMtx1Wy"
        "2GFqK012aKliVVa5VRBy2M57pMGXia9yHzezMSxT6cdDpeV6uzsWud6PPBPFAmgRSSqICiCqiKleSQqGteqE8mTdSwxwsMjE"
        "K6X6tlmTItS3RqyjGQ2RZd6kQxWGJXJytt2eZ3ajZLS17qWc3V6SrvjzzdqDSjOkMt+zjyNdBNz689bKY00ulFWvZ1Wl07UZ"
        "l22sRyC0iuWxrlVEFWxBVG4HQc/LxsL4vH7oa1hmo3o+e1u3n761z2z34TQzNlr2Yq9mkV7GaAoigAACKVqGuyynarsq4kcs"
        "qOYgtC6iUr+fUreqUopZa80RGyRkNALU9pbYpHuGq4hsiMqlV2GXOLFtxWYbNxFxJdyxnWNfvuxqGZyiKOsa5y2IqqjVVRBQ"
        "AAAE57oOcl4qJlXz+u1HCxLWvzvZdOV6zq4Po8/QS87rxaY9JqpYWpZpuzrUs5XZFtKEC6qYNU6dnIwHY1OVI30wpa0qqWJc"
        "+HoLEcmnaSnDy9spiOmh3mNrmw0A2JKy23WMizXsGUo1dZtzsmiJJUljVSVFVRFV9jHuVEVVEFEAAVAARRGw3MzIG6zLzW9m"
        "2eZQ9fHw9HJp2Eq8l6JV6RL9G8m88a7pmazkzakEuXR26E1lybljLm06qU5F3XPOUsdIphT66mbLdWqrrARLIIxXlNHCIyWI"
        "xYJoKY1zIQAvIyO29Pi6GbM2w6qEs+drO63Lts2jnJNZ3zM1M6RQmlVpY9WA8YDxgrxpCjUHkGeazeVzY7s85hPS2+aPPR18"
        "7fNehO4Cydjm4+XZ0VbktLeNabCoWemJ59axvs3c1auerWvazpooIKAAAAAAAAAAArQfA+nc5cCw2ua1iyFYysthLsmryRpa"
        "HPzG5Wz7bLrU8HTnzYqd/No9Bz3Q8e4qLjoEcgjmuEEBRAciYBoctkb2dY7O8t51wl7r3WczJ0aWc+u+tc+m+kuAm9Vs4avJ"
        "N289zcwOiXNp79TO8qnNs3NB25Q5dsvquc69mOzExLq0hbpVkJhqyqAAADQAAgkqawlWWt05ZcV6pNsfF0mdxEBjWJGkd3LJ"
        "WjjSkyN2VVqyWbtvAmRMvZxvT5NDouc6Pn0FQx0BUBzVEFBFEKXM6FzWJ7rl59gVblBQRQhrhaajiVsUyLw2R3/GXOJezKx0"
        "13j11nqdfkeo6culqWJuXavoZ08l5YiWVGKqRzKVEuCVZXxVMtJpeK1lQCCGYszqezV1jHj1KdWJZKeN0i2HEV7lvzezIk2H"
        "Z1mbFPP3jsKOdq9uUt7M27nPzxvp8Wj0fNdLz6ghjooAKgCtUWORq8ds5OP04+jup2+fZQIABQBFEAVA5+1xUWOm1bEvPch6"
        "TyHo4V7Ghal4al6Jy0upe4/sc6sWMmFOzOQ0sa3EpWrXIpI0eK1yNslSNCtJDWl1n1bVgCSox0Kwxyssa9ljOnCmb5xfzLPD"
        "16LklsjjsOsxrlmp052padz0eXOsVrHfy6PTcv0/Huo1cdFEBQQAUVFRcvg/T8e54PvOKxc32lfL+pXp1rWBVQAIyVua2Oa1"
        "aOzrOw2o+pOa6XN1nA08tnXht1sd+dZXofCdTx77mNv5S5GvaoxpV9Mly3abbKUz4YmKrV0WVGS07OXfL9mN9ggg2sq02FyR"
        "JM1+NgB5feq6Pn9caNxjoHwWJq2yrP056tSWH1eTOmhm9Piv9Py/T8e6qhjoooAKICgKCI4WridKHmWF7RVzfHtPtMRE1eWz"
        "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"
    ),
}


@app.route("/bg/<name>")
def background_image(name):

    data = BACKGROUND_IMAGES.get(name)

    if data is None:
        abort(404)

    return Response(
        data,
        mimetype="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


# =========================================================
# MENJALANKAN APLIKASI
# =========================================================

if __name__ == "__main__":

    # Untuk penggunaan lokal.
    # Untuk deployment, gunakan server WSGI.

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )