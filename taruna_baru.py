import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import ttest_ind
from datetime import datetime
import io
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Dashboard Keuangan Karang Taruna",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800;900&display=swap');

:root {
    --navy-dark:   #0F2044;
    --navy:        #1B3A6B;
    --navy-mid:    #2E5BBA;
    --navy-light:  #EEF3FB;
    --orange:      #F39C12;
    --orange-dark: #E67E22;
    --orange-light:#FEF5E7;
    --green:       #27AE60;
    --green-light: #E8F8EF;
    --red:         #E74C3C;
    --red-light:   #FDEDEC;
    --bg:          #F4F7FB;
    --card:        #FFFFFF;
    --border:      #DDE3EF;
    --text-dark:   #1A2D5A;
    --text-mid:    #4A5778;
    --text-light:  #8090B5;
}

*, html, body, [class*="css"] { font-family: 'Poppins', sans-serif !important; }
.stApp { background: var(--bg); }
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}

/* ======= HEADER ======= */
.main-header {
    background: linear-gradient(135deg, var(--navy-dark) 0%, var(--navy) 60%, var(--navy-mid) 100%);
    padding: 26px 36px;
    border-radius: 18px;
    margin-bottom: 22px;
    color: white;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(15,32,68,0.28);
    display: flex;
    align-items: center;
    gap: 20px;
}
.main-header::before {
    content: '';
    position: absolute; right: 0; top: 0;
    width: 320px; height: 100%;
    background: linear-gradient(135deg, transparent 40%, rgba(243,156,18,0.12) 100%);
    pointer-events: none;
}
.main-header::after {
    content: 'KT';
    position: absolute; right: 36px; top: 50%;
    transform: translateY(-50%);
    font-size: 5.5rem; font-weight: 900;
    opacity: 0.07; line-height: 1;
    color: var(--orange);
    letter-spacing: -4px;
    pointer-events: none;
}
.header-logo {
    width: 56px; height: 56px; border-radius: 16px;
    background: rgba(255,255,255,0.12);
    border: 2px solid rgba(255,255,255,0.25);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.65rem; flex-shrink: 0;
}
.header-badge-dot {
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    background: var(--orange); margin-right: 7px; vertical-align: middle;
}
.header-text h1 { font-size: 1.55rem; font-weight: 900; margin: 0 0 5px 0; line-height: 1.2; }
.header-text p  { font-size: 0.84rem; margin: 0; opacity: 0.75; }
.header-badge {
    margin-left: auto; flex-shrink: 0; z-index: 1;
    background: rgba(243,156,18,0.2);
    border: 1.5px solid rgba(243,156,18,0.45);
    border-radius: 30px; padding: 6px 16px;
    font-size: 0.76rem; font-weight: 700; color: #FDEBD0;
    white-space: nowrap;
}

/* ======= WELCOME SCREEN ======= */
.welcome-box {
    background: var(--card); border-radius: 20px; padding: 48px 36px;
    text-align: center; box-shadow: 0 4px 24px rgba(27,58,107,0.09);
    border: 2px dashed var(--border); margin-bottom: 20px;
}
.welcome-icon {
    width: 64px; height: 64px; border-radius: 20px; margin: 0 auto 14px;
    background: linear-gradient(135deg, #1B3A6B, #2E5BBA);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem; font-weight: 900; color: white; letter-spacing: 1px;
    box-shadow: 0 4px 18px rgba(46,91,186,0.35);
}
.welcome-title { font-size: 1.45rem; font-weight: 900; color: var(--navy); margin-bottom: 10px; }
.welcome-desc  { font-size: 0.90rem; color: var(--text-mid); line-height: 1.75; max-width: 500px; margin: 0 auto 28px; }
.welcome-steps { display: grid; grid-template-columns: repeat(3,1fr); gap: 14px; max-width: 640px; margin: 0 auto; text-align: left; }
.step-item { background: var(--navy-light); border-radius: 14px; padding: 16px; border: 1.5px solid #C9D6EE; }
.step-num  { background: var(--navy); color: white; border-radius: 50%; width: 26px; height: 26px; display: flex; align-items: center; justify-content: center; font-size: 0.76rem; font-weight: 800; margin-bottom: 9px; }
.step-title { font-size: 0.78rem; font-weight: 800; color: var(--navy); margin-bottom: 4px; }
.step-desc  { font-size: 0.72rem; color: var(--text-mid); line-height: 1.55; }

/* ======= SECTION TITLE ======= */
.section-title {
    font-size: 0.92rem; font-weight: 800; color: var(--navy);
    padding: 9px 16px; border-left: 5px solid var(--orange);
    background: var(--orange-light); border-radius: 0 10px 10px 0;
    margin: 18px 0 14px 0; display: block; line-height: 1.4;
}

/* ======= KPI CARDS ======= */
.kpi-card {
    background: var(--card); border-radius: 16px; padding: 20px 16px;
    text-align: center; box-shadow: 0 2px 14px rgba(27,58,107,0.08);
    border-top: 4px solid; height: 100%; box-sizing: border-box;
    overflow: hidden; transition: box-shadow 0.2s;
}
.kpi-card:hover { box-shadow: 0 6px 22px rgba(27,58,107,0.15); }
.kpi-label {
    font-size: 0.64rem; font-weight: 700; color: var(--text-light);
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; word-break: break-word;
}
.kpi-value { font-size: 1.18rem; font-weight: 900; color: var(--text-dark); line-height: 1.2; margin-bottom: 5px; }
.kpi-sub   { font-size: 0.66rem; color: var(--text-light); line-height: 1.4; }
.kpi-delta { font-size: 0.63rem; margin-top: 5px; font-weight: 700; }

/* ======= ANOMALI ======= */
.anomali-box {
    background: #FFF8E1; border: 2px solid var(--orange);
    border-radius: 12px; padding: 14px 16px; margin: 8px 0;
    display: flex; gap: 12px; align-items: flex-start;
}
.anomali-badge { font-size: 0.72rem; font-weight: 800; color: #7D3C00; background: #EDBB99; border-radius: 6px; padding: 3px 8px; flex-shrink: 0; line-height: 1.6; }
.anomali-title { font-size: 0.82rem; font-weight: 800; color: #7D3C00; margin-bottom: 3px; }
.anomali-text  { font-size: 0.73rem; color: #7D4E00; line-height: 1.55; }

/* ======= PREDIKSI BOX ======= */
.pred-box {
    background: linear-gradient(135deg, var(--navy-dark) 0%, var(--navy) 60%, var(--navy-mid) 100%);
    border-radius: 18px; padding: 26px 28px; color: white;
    box-shadow: 0 8px 28px rgba(15,32,68,0.28);
    position: relative; overflow: hidden; margin-bottom: 18px;
}
.pred-box-title { font-size: 1.12rem; font-weight: 800; margin-bottom: 5px; }
.pred-box-sub   {
    font-size: 0.80rem; opacity: 0.75; margin-bottom: 20px;
    background: rgba(255,255,255,0.07); border-radius: 8px; padding: 6px 12px; display: inline-block;
}
.pred-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; }
.pred-item {
    background: rgba(255,255,255,0.10); border: 1.5px solid rgba(255,255,255,0.18);
    border-radius: 13px; padding: 14px 10px; text-align: center;
}
.pred-item-label { font-size: 0.68rem; opacity: 0.80; font-weight: 600; margin-bottom: 8px; }
.pred-item-value { font-size: 1.02rem; font-weight: 900; line-height: 1.2; }
.pred-item-delta { font-size: 0.64rem; opacity: 0.70; margin-top: 4px; }

/* ======= FORECAST CARDS ======= */
.fc-card { background: var(--card); border-radius: 14px; padding: 18px 14px; text-align: center; box-shadow: 0 2px 12px rgba(27,58,107,0.08); border-top: 4px solid var(--navy); flex: 1; }
.fc-month { font-size: 0.70rem; font-weight: 700; color: var(--text-light); text-transform: uppercase; margin-bottom: 8px; }
.fc-value { font-size: 1.05rem; font-weight: 900; color: var(--navy); margin-bottom: 3px; }
.fc-ci    { font-size: 0.65rem; color: var(--text-light); }

/* ======= HEALTH BARS ======= */
.health-row { margin-bottom: 20px; }
.health-meta { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.health-label { font-size: 0.84rem; font-weight: 700; color: var(--text-dark); }
.health-score { font-size: 0.84rem; font-weight: 900; }
.health-bar-bg   { background: var(--border); border-radius: 10px; height: 12px; overflow: hidden; }
.health-bar-fill { height: 100%; border-radius: 10px; transition: width 0.6s ease; }
.health-sub { font-size: 0.70rem; color: var(--text-light); margin-top: 5px; }

/* ======= SCORE CIRCLE ======= */
.score-circle {
    border-radius: 18px; padding: 24px 14px; text-align: center; height: 100%;
    display: flex; flex-direction: column; justify-content: center; align-items: center;
    min-height: 190px;
}
.score-num   { font-size: 2.8rem; font-weight: 900; line-height: 1; }
.score-label { font-size: 0.88rem; font-weight: 800; margin-top: 8px; }
.score-sub   { font-size: 0.68rem; color: var(--text-mid); margin-top: 8px; line-height: 1.4; text-align: center; }

/* ======= TIP CARDS ======= */
.tip-card {
    background: var(--card); border-radius: 14px; padding: 16px;
    box-shadow: 0 2px 10px rgba(27,58,107,0.07); border-left: 5px solid;
    display: flex; gap: 12px; align-items: flex-start; margin-bottom: 12px;
}
.tip-badge { width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 0.72rem; font-weight: 900; color: white; }
.tip-title { font-size: 0.82rem; font-weight: 800; color: var(--text-dark); margin-bottom: 4px; }
.tip-text  { font-size: 0.73rem; color: var(--text-mid); line-height: 1.60; }

/* ======= A/B CARDS ======= */
.ab-card { border-radius: 14px; padding: 20px 16px; text-align: center; }
.ab-card.awal  { background: var(--green-light); border: 2px solid #A9DFBF; }
.ab-card.akhir { background: var(--navy-light);  border: 2px solid #C9D6EE; }
.ab-period { font-size: 0.78rem; font-weight: 700; color: var(--text-dark); margin-bottom: 8px; }
.ab-value  { font-size: 1.38rem; font-weight: 900; margin-bottom: 5px; }
.ab-card.awal  .ab-value { color: var(--green); }
.ab-card.akhir .ab-value { color: var(--navy); }
.ab-desc { font-size: 0.68rem; color: var(--text-mid); }
.ab-card.warning { background: var(--orange-light); border: 2px solid var(--orange); }
.ab-card.warning .ab-value { color: var(--orange-dark); }

/* ======= DIVIDER ======= */
.custom-divider {
    height: 2px; background: linear-gradient(90deg, var(--orange) 0%, transparent 100%);
    border: none; border-radius: 2px; margin: 18px 0;
}

/* ======= FOOTER ======= */
.footer-box {
    background: linear-gradient(135deg, var(--navy-dark), var(--navy));
    color: white; text-align: center; padding: 20px; border-radius: 16px;
    margin-top: 30px; font-size: 0.80rem; border-top: 3px solid var(--orange);
}
.footer-box span { opacity: 0.60; font-size: 0.70rem; display: block; margin-top: 5px; }

/* ======= DOWNLOAD BUTTON ======= */
.stDownloadButton button {
    background: var(--navy) !important; color: white !important;
    border-radius: 10px !important; font-weight: 700 !important;
    border: none !important; width: 100% !important; transition: background 0.2s !important;
}
.stDownloadButton button:hover { background: var(--navy-dark) !important; }

/* ======= TABS ======= */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background: var(--card); border-radius: 14px;
    padding: 6px 8px; box-shadow: 0 2px 12px rgba(27,58,107,0.08);
    margin-bottom: 18px; border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important; padding: 8px 16px !important;
    font-weight: 700 !important; font-size: 0.82rem !important; color: var(--text-mid) !important;
}
.stTabs [aria-selected="true"] { background: var(--navy) !important; color: white !important; }

/* ======= SIDEBAR ======= */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1A3A 0%, #0F2044 60%, #132652 100%) !important;
    border-right: 1px solid rgba(243,156,18,0.18) !important;
}
[data-testid="stSidebar"] > div { padding-top: 0.5rem !important; padding-bottom: 2rem !important; }

[data-testid="stSidebarCollapseButton"] {
    display: flex !important; visibility: visible !important;
    background: rgba(243,156,18,0.15) !important;
    border: 1px solid rgba(243,156,18,0.35) !important;
    border-radius: 8px !important;
    font-size: 0 !important;
    color: transparent !important;
    overflow: hidden !important;
}
[data-testid="stSidebarCollapseButton"] * {
    font-size: 0 !important;
    color: transparent !important;
}
[data-testid="stSidebarCollapseButton"] svg {
    fill: var(--orange) !important;
    width: 18px !important; height: 18px !important;
    flex-shrink: 0 !important;
    font-size: initial !important;
}

/* Kotak navigasi saat sidebar ditutup — tambah ikon panah */
[data-testid="collapsedControl"] {
    display: flex !important; visibility: visible !important;
    align-items: center !important; justify-content: center !important;
    background: var(--navy) !important;
    border: 1px solid rgba(243,156,18,0.45) !important;
    border-radius: 0 10px 10px 0 !important;
    width: 28px !important; min-width: 28px !important;
    font-size: 0 !important; color: transparent !important;
    overflow: hidden !important;
    box-shadow: 3px 0 12px rgba(0,0,0,0.18) !important;
}
[data-testid="collapsedControl"] * {
    font-size: 0 !important;
    color: transparent !important;
}
[data-testid="collapsedControl"] svg {
    fill: var(--orange) !important;
    width: 18px !important; height: 18px !important;
    flex-shrink: 0 !important;
    font-size: initial !important;
}
/* Ikon panah kanan via pseudo-element sebagai fallback */
[data-testid="collapsedControl"]::after {
    content: '›';
    color: var(--orange) !important;
    font-size: 1.5rem !important;
    font-weight: 900 !important;
    line-height: 1 !important;
    display: block !important;
}

/* Label selectbox — Pilih Tahun & Pilih Bulan */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSelectbox > label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 0.84rem !important;
    letter-spacing: 0.2px !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stMarkdown { color: rgba(255,255,255,0.85) !important; }
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important; font-weight: 800 !important;
    font-size: 0.92rem !important; margin-bottom: 10px !important; letter-spacing: 0.3px !important;
}

/* SELECT BOX */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: background-color: #1B3A6B !important;
    border: 1.5px solid rgba(255,255,255,0.60) !important;
    border-radius: 10px !important; transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div:hover {
    Profesionalisasi dashboard StreamlitClaude finished the responseYou said: ubah tampilan dari dashboard streamlit menjadi lebih profesional dengan mengurangi simbol yang tidak pentingdashboard_karang_taruna.pypyData Sintesis (CP).xlsxxlsxrequirements.txt8 linestxtubah tampilan dari dashboard streamlit menjadi lebih profesional dengan mengurangi simbol yang tidak penting15 MeiClaude responded: Sekarang saya sudah memahami keseluruhan kode.Merancang penghapusan simbol untuk profesionalisasi dashboard StreamlitMerancang penghapusan simbol untuk profesionalisasi dashboard StreamlitSekarang saya sudah memahami keseluruhan kode. Saya akan menyalin file dan membersihkan semua simbol/emoji yang tidak perlu untuk tampilan yang lebih profesional.Edited 21 files, ran a commandEdited 21 files, ran a commandYou said: mungkin ini bisa diganti jadi lebih rapi dan menggunakan sidebar yang interaktif, kemudian tulisan tahun dan bulan dibuat lebih jelas saat dibacamungkin ini bisa diganti jadi lebih rapi dan menggunakan sidebar yang interaktif, kemudian tulisan tahun dan bulan dibuat lebih jelas saat dibaca16 MeiClaude responded: Saya lihat masalahnya — ada dua tombol "Pilih File Excel" yang muncul dan teks dropdown tahun/bulan gelap sehingga sulit dibaca.Mengidentifikasi masalah sidebar dan merencanakan perbaikan CSSMengidentifikasi masalah sidebar dan merencanakan perbaikan CSSSaya lihat masalahnya — ada dua tombol "Pilih File Excel" yang muncul dan teks dropdown tahun/bulan gelap sehingga sulit dibaca. Saya akan perbaiki CSS sidebar secara menyeluruh.Overhaul sidebar CSS - fix selectbox readability, remove duplicate button, make interactiveOverhaul sidebar CSS - fix selectbox readability, remove duplicate button, make interactiveSekarang perbarui CSS logo sidebar agar teks "KT" tampil dengan baik:Update sidebar logo CSS to display KT text properlyUpdate sidebar logo CSS to display KT text properlySekarang tambahkan CSS untuk error dan warning agar lebih bersih di sidebar:Edited 4 files, viewed 2 filesEdited 4 files, viewed 2 filesSekarang bersihkan sisa emoji di chart titles dan status kolom:Systematized emoji removal across interface components comprehensivelySystematized emoji removal across interface components comprehensivelySekarang bersihkan sisa emoji di chart titles, tabel, prediksi box, dan tips:Ran 4 commands, viewed a file, edited a fileRan 4 commands, viewed a file, edited a fileSekarang lakukan uji sintaks Python dan salin ke output:Mengidentifikasi dan memperbaiki kesalahan sintaks kamus yang rusakMengidentifikasi dan memperbaiki kesalahan sintaks kamus yang rusakPerbaiki sintaks dictionary yang rusak akibat replace icon:Edited a file, ran a commandEdited a file, ran a commandSintaks OK. Sekarang rapikan tampilan tip-card agar icon teks terlihat profesional:Viewed a file, edited a file, ran a commandViewed a file, edited a file, ran a commandBerikut ringkasan semua perubahan yang dilakukan:
Sidebar (fokus utama):

Tombol collapse/expand sidebar sekarang ditampilkan kembali (sebelumnya disembunyikan) sehingga sidebar bisa dibuka-tutup secara interaktif
Selectbox Tahun & Bulan — teks nilai yang dipilih kini putih tebal (font-weight: 700) dan kontras tinggi, tidak lagi gelap/sulit dibaca
Dropdown menu saat terbuka juga disesuaikan warnanya (background navy, teks putih, hover orange)
Tombol upload — dihilangkan duplikasi "Pilih File Excel" ganda; sekarang hanya satu tombol bersih
File yang sudah diupload ditampilkan dengan background hijau muda agar terlihat jelas statusnya
Semua label (Pilih Tahun, Pilih Bulan, Filter Tampilan) lebih terang dan tebal

Seluruh dashboard:

Semua emoji dekoratif dihapus dari tab, judul seksi, header, footer, grafik, tabel, badge status, pesan error/warning
Ikon tip-card diganti dengan lingkaran berwarna (merah/hijau/biru/kuning) sesuai konteks
Indikator kepercayaan prediksi diganti dengan titik warna (merah/kuning/hijau)
Kolom tabel (Bulan, Pemasukan, dll.) tampil tanpa emoji
Dashboard karang tarunaPY DownloadYou said: perbaikan sidebar yang aku inginkan ada di kode yang manaperbaikan sidebar yang aku inginkan ada di kode yang mana16 MeiClaude responded: Perbaikan sidebar tersebar di 3 bagian dalam kode:Mengidentifikasi lokasi perbaikan sidebar dalam kodeMengidentifikasi lokasi perbaikan sidebar dalam kodePerbaikan sidebar tersebar di 3 bagian dalam kode:

1. CSS Sidebar — baris ~336 s/d ~560
Ini yang mengatur tampilan visual seluruh sidebar:

~336–430 → warna background, tombol collapse, label, selectbox (termasuk warna teks Tahun/Bulan)
~430–500 → file uploader (tombol upload, file yang sudah dipilih)
~500–560 → kartu info, badge surplus/defisit, divider, logo, menu navigasi, footer


2. Fungsi render_sidebar() — sekitar baris ~1018
Ini yang mengatur konten/logika sidebar (bukan tampilan):

Logo KT di atas
Widget st.file_uploader
Selectbox Pilih Tahun dan Pilih Bulan
Kartu ringkasan data (jumlah transaksi, pemasukan, pengeluaran)
Menu navigasi
Footer


3. CSS dropdown menu global — dalam blok CSS yang sama (~baris 431–445)
css[data-baseweb="popover"] [data-baseweb="menu"] { ... }
[data-baseweb="popover"] [role="option"] { ... }
Ini khusus untuk warna teks saat dropdown terbuka (pilihan tahun/bulan yang muncul ke bawah) — perlu selector terpisah karena dropdown Streamlit render di luar elemen sidebar secara teknis.

Kalau mau cek langsung, cari teks ini di editor:

/* ======= SIDEBAR ======= untuk blok CSS
def render_sidebar(): untuk fungsi kontennya
You said: kode untuk hapus ikon bagian ringkasan yang mana?kode untuk hapus ikon bagian ringkasan yang mana?16 MeiClaude responded: Bagian yang menghapus ikon di Ringkasan ada di beberapa tempat:Bagian yang menghapus ikon di Ringkasan ada di beberapa tempat:

1. KPI Cards (kartu angka utama) — fungsi render_kpi() sekitar baris ~1185
pythonkartu = [
    {'w': CLR_NAVY,  'icon':'',   # ← dikosongkan (aslinya '🏦')
     'label':'Saldo Kas Sekarang', ...},
    {'w': CLR_GREEN, 'icon':'',   # ← dikosongkan (aslinya '📥')
     'label':'Total Pemasukan', ...},
    ...
]

2. Judul seksi Ringkasan — di fungsi main() sekitar baris ~2039
pythonst.markdown(
    f'<div class="section-title">Ringkasan Keuangan — Tahun {tahun_aktif}...',
    # aslinya: '📌 Ringkasan Keuangan — ...'
)

st.markdown(
    '<div class="section-title">Kondisi Kesehatan Keuangan</div>',
    # aslinya: '🏥 Kondisi Kesehatan Keuangan'
)

3. Tab label — di fungsi main() sekitar baris ~2027
pythontab1, tab2, ... = st.tabs([
    "Ringkasan",   # aslinya: "📌 Ringkasan"
    "Grafik",      # aslinya: "📈 Grafik"
    ...
])

4. Caption health score — sekitar baris ~2051
pythonst.caption(
    "Skor 0–100%  ·  Sangat Sehat (70–100%)  ·  ..."
    # aslinya ada 🟢 🟡 🔴 di depan setiap kategori
)
Intinya semua penghapusan ikon di Ringkasan dilakukan dengan mengosongkan string atau menghapus karakter emoji langsung dari nilai yang dikirim ke HTML/markdown.You said: hapuslah sebagian ikon dari kode tersebut sehingga tampilannya sangat profesionalimport streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import ttest_ind
from datetimpastedhapuslah sebagian ikon dari kode tersebut sehingga tampilannya sangat profesional16 MeiClaude responded: Baik, saya akan bekerja langsung dari kode yang dikirim dan membuat versi bersih yang profesional secara menyeluruh.Systematisasi penghapusan ikon untuk menciptakan desain profesionalSystematisasi penghapusan ikon untuk menciptakan desain profesionalBaik, saya akan bekerja langsung dari kode yang dikirim dan membuat versi bersih yang profesional secara menyeluruh.Write the complete clean professional version from scratchWrite the complete clean professional version from scratchWrite the complete clean professional version from scratchScriptDoneYou said: lanjutkanlanjutkan13.35Claude responded: Selesai.Assessed syntax dan memindahkan file ke outputAssessed syntax dan memindahkan file ke outputSelesai. Ini versi bersih yang ditulis ulang penuh dari kode asli. Berikut ringkasan semua yang diubah:
Ikon dihapus sepenuhnya dari:

Tab navigasi (Ringkasan, Grafik, Prediksi, Detail, Tips & Saran)
Semua judul seksi (section-title)
Header utama dan badge — diganti teks KT
Sidebar: logo, label filter, kartu ringkasan, menu navigasi, footer
KPI cards — kolom icon dihapus, padding disesuaikan
Prediksi box dan 4 item di dalamnya
Kartu forecast 3 bulan
Ringkasan prediksi dan pesan kondisi keuangan
Anomali — diganti badge teks Naik / Turun
Semua judul grafik (Plotly chart titles)
Nama legenda grafik bar (Pemasukan, Pengeluaran)
Kolom tabel rekap
Tombol download
Pesan error, warning, info, dan success
Footer

Diganti dengan alternatif visual yang bersih:

Indikator kesehatan keuangan → titik warna (dot CSS) bukan emoji
Skor kesehatan keseluruhan → angka dan label teks saja, tanpa face emoji
Tips card → lingkaran berwarna dengan label singkat (!, OK, #1, i, N)
Akurasi prediksi → titik warna sesuai level kepercayaan (hijau/kuning/merah)
Status surplus/defisit di tabel → teks bersih tanpa simbol
Dashboard karang tarunaPY DownloadYou said: gantilah icon seperlunya saja, untuk header utama dan badge jangan hanya tulisan, itu kurang profesional.gantilah icon seperlunya saja, untuk header utama dan badge jangan hanya tulisan, itu kurang profesional.13.37Claude responded: Saya akan tambahkan ikon kembali secara selektif — fokus di header, badge, dan beberapa titik yang memang membantu secara visual.Strategized targeted icon placement untuk header dan badge profesionalStrategized targeted icon placement untuk header dan badge profesionalSaya akan tambahkan ikon kembali secara selektif — fokus di header, badge, dan beberapa titik yang memang membantu secara visual.Viewed 2 files, edited 2 filesViewed 2 files, edited 2 filesSekarang update konten header — logo pakai SVG inline, badge dengan indikator visual:Update main header with SVG icon and better badgeUpdate main header with SVG icon and better badgeSekarang tambahkan ikon SVG juga di sidebar logo dan welcome screen:Edited 2 filesEdited 2 filesSekarang tambahkan beberapa ikon fungsional di tempat yang membantu — KPI cards dan tab:Edited 2 files, ran a command, read a fileEdited 2 files, ran a command, read a fileIni yang diubah:
Header utama — logo diganti SVG ikon gedung/institusi (bukan emoji, bukan tulisan), badge sekarang punya titik oranye kecil di depan teks "Portal Manajemen Desa" sehingga terasa lebih bernyawa.
Sidebar — logo KT di atas juga pakai SVG ikon gedung yang sama, konsisten dengan header.
Welcome screen — ikon di tengah juga SVG gedung, bukan teks "KT".
KPI cards — ditambahkan ikon minimal yang fungsional: 🏦 untuk Saldo Kas, ↑ / ↓ untuk Pemasukan/Pengeluaran, ▲ / ▼ untuk Surplus/Defisit (berubah warna sesuai kondisi), % untuk Persentase Ditabung — semuanya berwarna sesuai tema kartu, bukan dekoratif.
Bagian lain (tab, section title, tabel, grafik, tips, footer) tetap bersih tanpa ikon karena tidak dibutuhkan secara fungsional.Dashboard karang tarunaPY DownloadYou said: tampilkan kode bagian filter tampilan bulan dan tahuntampilkan kode bagian filter tampilan bulan dan tahun13.40Claude responded: Ini kodenya, baris 692–698:Ran a command, viewed a fileRan a command, viewed a fileIni kodenya, baris 692–698:
pythonst.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
st.markdown("### Filter Tampilan")

tahun_list = sorted(df_preview['Tahun'].unique(), reverse=True)
tahun_sel  = st.selectbox("Pilih Tahun", options=tahun_list, format_func=lambda x: f"Tahun {x}")

bulan_tersedia = sorted(df_preview[df_preview['Tahun'] == tahun_sel]['Bulan'].unique())
bulan_opts = ['Semua Bulan'] + [MONTH_ID[b] for b in bulan_tersedia]
bulan_sel  = st.selectbox("Pilih Bulan", options=bulan_opts)
Penjelasan singkat tiap baris:

tahun_list — ambil semua tahun unik dari data, diurutkan terbaru di atas
tahun_sel — selectbox Tahun, nilai ditampilkan sebagai "Tahun 2024" lewat format_func
bulan_tersedia — hanya bulan yang ada datanya di tahun yang dipilih (bukan semua 12 bulan)
bulan_opts — gabungkan opsi "Semua Bulan" + nama bulan dalam Bahasa Indonesia (MONTH_ID)
bulan_sel — selectbox Bulan, nilai balik berupa nama bulan seperti "Januari", "Februari", dst.

Ada yang ingin diubah dari bagian ini?You said: apakah bisa warna fontnya diganti?apakah bisa warna fontnya diganti?13.41Claude responded: Bisa.Bisa. Font color di sidebar bisa diubah lewat CSS yang sudah ada. Mau diganti seperti apa?Silakan pilih dulu ya — bagian mana dan warna apa yang diinginkan, nanti langsung saya terapkan ke kode.You said: Q: Bagian mana yang ingin diganti warna fontnya?Q: Bagian mana yang ingin diganti warna fontnya?
A: KeduanyaQ: Warna yang diinginkan?
A: Putih penuh (#FFFFFF)13.42Claude responded: Keduanya sudah saya terapkan.Memverifikasi CSS dropdown dan label sudah berwarna putihMemverifikasi CSS dropdown dan label sudah berwarna putihKeduanya sudah saya terapkan. Mari pastikan CSS-nya sudah mencakup semua kondisi dengan benar:Reinforced CSS selectors untuk memastikan font putih penuhReinforced CSS selectors untuk memastikan font putih penuhLabel dan nilai sudah diset #FFFFFF, tapi perlu diperkuat agar tidak tertimpa CSS bawaan Streamlit. Saya tambahkan selector yang lebih lengkap:Edited a file, ran a command, read a fileEdited a file, ran a command, read a filePerubahan yang dilakukan — CSS diperkuat dengan selector tambahan:
Label (Pilih Tahun & Pilih Bulan):

Ditambah label p, .stSelectbox label, [data-testid="stWidgetLabel"], dan [data-testid="stWidgetLabel"] p — karena Streamlit terkadang membungkus teks label dalam elemen yang berbeda tergantung versi
Ditambah opacity: 1 !important agar tidak ada transparansi tersembunyi

Nilai terpilih di dalam kotak dropdown:

Ditambah selector input dan [class*="Input"] untuk menangkap kasus di mana nilai dirender sebagai input field
Ditambah opacity: 1 !important untuk memastikan penuh putih tanpa fade
Dashboard karang tarunaPY DownloadYou said: gantilah warna dari icon inigantilah warna dari icon ini13.50Claude responded: Pilih warna yang diinginkan ya, nanti langsung saya terapkan.Deliberated icon color modification without specified target hueDeliberated icon color modification without specified target huePilih warna yang diinginkan ya, nanti langsung saya terapkan.You said: Q: Mau diganti ke warna apa?Q: Mau diganti ke warna apa?
A: Biru terang (#2E5BBA)13.50Claude responded: Selesai.Melacak lokasi CSS sidebar logo untuk perubahan warnaMelacak lokasi CSS sidebar logo untuk perubahan warnaSelesai. Background icon berubah dari gradien oranye menjadi gradien navy → biru terang (#1B3A6B ke #2E5BBA), shadow-nya juga disesuaikan ke warna biru. Hasilnya akan lebih serasi dengan tema header utama dashboard.Dashboard karang tarunaPY DownloadYou said: mau lihat kode untuk mengganti warna tersebut, agar aku dapat menggantinya sendiri.mau lihat kode untuk mengganti warna tersebut, agar aku dapat menggantinya sendiri. berikan langkah-langkah dalam pergantiannya13.52Claude responded: Berikut kode yang mengatur warna icon tersebut, ada di bagian CSS sekitar baris ~400:Berikut kode yang mengatur warna icon tersebut, ada di bagian CSS sekitar baris ~400:
css.sidebar-logo .logo-wrap {
    background: linear-gradient(135deg, #1B3A6B, #2E5BBA);
    box-shadow: 0 4px 18px rgba(46,91,186,0.40);
}

Langkah-langkah menggantinya:
1. Buka file dashboard_karang_taruna.py di editor (VS Code, Notepad++, dll.)
2. Cari teks .sidebar-logo .logo-wrap menggunakan Ctrl+F
3. Ganti warna background — ada dua angka hex di baris background:
cssbackground: linear-gradient(135deg, #1B3A6B, #2E5BBA);
/*                                   ↑ warna awal   ↑ warna akhir */
Misalnya ingin oranye kembali:
cssbackground: linear-gradient(135deg, #E67E22, #F39C12);
4. Ganti warna shadow — sesuaikan rgba di baris box-shadow agar cocok dengan warna baru:
cssbox-shadow: 0 4px 18px rgba(46,91,186,0.40);
/*                          ↑ ini adalah versi RGB dari warna utama */
Untuk oranye misalnya:
cssbox-shadow: 0 4px 18px rgba(243,156,18,0.40);
5. Simpan file lalu jalankan ulang Streamlit.

Referensi konversi hex ke RGB (untuk shadow):
HexRGB#1B3A6B27, 58, 107#2E5BBA46, 91, 186#F39C12243, 156, 18#27AE6039, 174, 96#E74C3C231, 76, 60
Angka 0.40 di akhir rgba adalah tingkat transparansi shadow (0.0 = transparan, 1.0 = penuh). Bisa dibiarkan tetap 0.40 untuk semua warna.You said: untuk mengganti warna filter bulan dan tahun agar menjadi tidak transparan caranya gimana?untuk mengganti warna filter bulan dan tahun agar menjadi tidak transparan caranya gimana?13.56Claude responded: Cari bagian ini di CSS sekitar baris ~294, gunakan Ctrl+F dengan kata kunci .Cari bagian ini di CSS sekitar baris ~294, gunakan Ctrl+F dengan kata kunci .sidebar-logo .logo-wrap lalu scroll sedikit ke bawah, atau cari langsung /* SELECT BOX */:
css/* SELECT BOX */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.12) !important;  /* ← ini yang bikin transparan */
    border: 1.5px solid rgba(255,255,255,0.30) !important;
    border-radius: 10px !important;
}

Langkah-langkah:
1. Cari teks /* SELECT BOX */ dengan Ctrl+F
2. Ganti background-color — hilangkan transparansi:
css/* Sebelum (transparan 12%) */
background-color: rgba(255,255,255,0.12) !important;

/* Sesudah (tidak transparan, warna solid) */
background-color: #1B3A6B !important;
3. Ganti juga border agar sesuai:
css/* Sebelum */
border: 1.5px solid rgba(255,255,255,0.30) !important;

/* Sesudah */
border: 1.5px solid rgba(255,255,255,0.60) !important;
4. Ganti juga efek hover di baris berikutnya:
css[data-testid="stSidebar"] [data-baseweb="select"] > div:hover {
    background-color: rgba(255,255,255,0.18) !important; /* ← ini juga transparan */
}
Menjadi:
cssbackground-color: #2E5BBA !important; border-color: var(--orange) !important;
}
/* Nilai terpilih di dalam dropdown */
[data-testid="stSidebar"] [data-baseweb="select"] [class*="singleValue"],
[data-testid="stSidebar"] [data-baseweb="select"] [class*="placeholder"],
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] div[class*="ValueContainer"] > div,
[data-testid="stSidebar"] [data-baseweb="select"] input,
[data-testid="stSidebar"] [data-baseweb="select"] [class*="Input"] {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: rgba(255,255,255,0.75) !important; }
[data-baseweb="popover"] [data-baseweb="menu"] {
    background-color: #FFFFFF  !important;
    border: 1px solid #DDE3EF !important; border-radius: 10px !important;
}
[data-baseweb="popover"] [role="option"] {
    background-color: transparent !important; color: #1B3A6B  !important; font-size: 0.85rem !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] {
    background-color: rgba(243,156,18,0.20) !important; color: #1B3A6B !important;
}

/* FILE UPLOADER */
[data-testid="stSidebar"] [data-testid="stFileUploader"] > label { display: none !important; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.06) !important;
    border: 2px dashed rgba(243,156,18,0.50) !important;
    border-radius: 12px !important; padding: 20px 14px !important;
    text-align: center !important; transition: all 0.2s !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover {
    background: rgba(243,156,18,0.08) !important; border-color: var(--orange) !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] p,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {
    color: rgba(255,255,255,0.80) !important; font-size: 0.80rem !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
    background: var(--orange-dark) !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    padding: 8px 20px !important; margin-top: 8px !important;
    font-size: 0.82rem !important; font-weight: 700 !important;
    font-family: 'Poppins', sans-serif !important; cursor: pointer !important;
    width: 100% !important; transition: background 0.2s !important;
    overflow: hidden !important;
}
/* Sembunyikan span duplikat penyebab teks "uploadupload" */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button span:first-child {
    display: none !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button span:last-child {
    display: inline !important;
    color: white !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {
    background: rgba(39,174,96,0.12) !important; border-radius: 8px !important;
    padding: 10px 12px !important; border: 1px solid rgba(39,174,96,0.35) !important; margin-top: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] span,
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] p,
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] small,
[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"] {
    color: #A9DFBF !important; opacity: 1 !important; font-size: 0.80rem !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] svg { fill: #27AE60 !important; }

/* Sidebar Components */
.sidebar-info {
    background: rgba(255,255,255,0.07); border-radius: 12px; padding: 16px;
    font-size: 0.80rem; color: rgba(255,255,255,0.90); line-height: 1.85;
    border: 1px solid rgba(255,255,255,0.12);
}
.sidebar-info strong { color: #FDEBD0; }
.sidebar-badge {
    display: inline-block; border-radius: 20px; padding: 4px 14px;
    font-size: 0.72rem; font-weight: 700; margin-top: 8px; letter-spacing: 0.3px;
}
.sidebar-badge.surplus { background: rgba(39,174,96,0.25); color: #A9DFBF; border: 1px solid rgba(39,174,96,0.4); }
.sidebar-badge.defisit { background: rgba(231,76,60,0.25); color: #F1948A; border: 1px solid rgba(231,76,60,0.4); }
.sidebar-divider { border: none; border-top: 1px solid rgba(255,255,255,0.12); margin: 16px 0; }
.sidebar-logo { text-align: center; padding: 20px 0 18px; border-bottom: 1px solid rgba(243,156,18,0.25); margin-bottom: 18px; }
.sidebar-logo .logo-wrap {
    display: inline-flex; align-items: center; justify-content: center;
    width: 60px; height: 60px; border-radius: 18px;
    background: linear-gradient(135deg, #1B3A6B, #2E5BBA);
    font-size: 1.3rem; font-weight: 900; color: white; margin-bottom: 10px;
    box-shadow: 0 4px 18px rgba(46,91,186,0.40); letter-spacing: 1px;
}
.sidebar-logo .logo-title { font-size: 1.05rem; font-weight: 900; color: white; display: block; letter-spacing: 0.3px; }
.sidebar-logo .logo-sub   { font-size: 0.74rem; color: rgba(243,156,18,0.85); font-weight: 500; display: block; margin-top: 3px; }
.sidebar-nav {
    background: rgba(255,255,255,0.06); border-radius: 12px; padding: 14px 16px;
    margin-top: 12px; font-size: 0.78rem; color: rgba(255,255,255,0.80);
    line-height: 2.4; border: 1px solid rgba(255,255,255,0.10);
}
.sidebar-nav strong { color: white; }
.sidebar-footer { margin-top: 22px; text-align: center; font-size: 0.70rem; color: rgba(243,156,18,0.55); padding-bottom: 8px; }

/* YoY box */
.yoy-box { background: var(--card); border-radius: 12px; padding: 16px; box-shadow: 0 2px 10px rgba(27,58,107,0.07); border-left: 4px solid var(--navy-mid); margin-bottom: 12px; }
.yoy-title { font-size: 0.82rem; font-weight: 800; color: var(--text-dark); margin-bottom: 8px; }
.yoy-val   { font-size: 1.10rem; font-weight: 900; }
.yoy-delta { font-size: 0.70rem; font-weight: 700; }

/* Metric badge row */
.metric-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--navy-light); border-radius: 8px;
    padding: 5px 12px; font-size: 0.74rem; font-weight: 700;
    color: var(--navy); margin: 4px 4px 0 0; border: 1px solid #C9D6EE;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# KONSTANTA & HELPERS
# ============================================================

MONTH_ID = {
    1:"Januari", 2:"Februari", 3:"Maret",    4:"April",
    5:"Mei",     6:"Juni",     7:"Juli",      8:"Agustus",
    9:"September",10:"Oktober",11:"November",12:"Desember"
}
MONTH_SHORT = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"Mei",6:"Jun",7:"Jul",8:"Agu",9:"Sep",10:"Okt",11:"Nov",12:"Des"}
CLR_GREEN  = "#27AE60"; CLR_RED  = "#E74C3C"; CLR_BLUE  = "#2E5BBA"
CLR_GOLD   = "#F39C12"; CLR_DARK = "#1B3A6B"; CLR_NAVY  = "#1B3A6B"
LAYOUT = dict(font=dict(family="Poppins, sans-serif", size=11, color="#1A2D5A"),
              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
              margin=dict(l=12, r=12, t=64, b=16))

def rupiah(n, singkat=False):
    try:
        n = float(n)
        if np.isnan(n): n = 0.0
    except Exception: n = 0.0
    n = int(n); negatif = n < 0; absn = abs(n)
    if singkat:
        if absn >= 1_000_000_000: teks = f"Rp {absn/1_000_000_000:.1f} M"
        elif absn >= 1_000_000:   teks = f"Rp {absn/1_000_000:.1f} jt"
        elif absn >= 1_000:       teks = f"Rp {absn/1_000:.0f} rb"
        else:                      teks = "Rp " + f"{absn:,}".replace(",",".")
    else:
        teks = "Rp " + f"{absn:,}".replace(",",".")
    return ("-" + teks) if negatif else teks

def clr_health(v): return CLR_GREEN if v >= 70 else ("#F39C12" if v >= 40 else CLR_RED)
def dot_health(v):
    c = clr_health(v)
    return f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{c};vertical-align:middle;margin-right:5px;"></span>'

# ============================================================
# BERSIHKAN DATA
# ============================================================

def bersihkan_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().dropna(how='all').reset_index(drop=True)
    df.columns = df.columns.str.strip()
    col_map = {}
    for col in df.columns:
        c = col.lower().strip()
        if   any(k in c for k in ['tanggal','date','tgl']):            col_map.setdefault('Tanggal',    col)
        elif any(k in c for k in ['pemasukan','income','masuk','debit_masuk','kredit_masuk']): col_map.setdefault('Pemasukan',  col)
        elif any(k in c for k in ['pengeluaran','expense','keluar','kredit','bayar','debit_keluar']): col_map.setdefault('Pengeluaran',col)
        elif any(k in c for k in ['saldo','balance','kas','neraca']):   col_map.setdefault('Saldo',      col)
        elif any(k in c for k in ['keterangan','ket','description','catatan','uraian','remark']): col_map.setdefault('Keterangan', col)
        elif any(k in c for k in ['kategori','category','jenis','type']): col_map.setdefault('Kategori',   col)
    df = df.rename(columns={v: k for k, v in col_map.items()})
    missing = [c for c in ['Tanggal','Pemasukan','Pengeluaran'] if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: **{', '.join(missing)}**\n\nKolom tersedia: {', '.join(df.columns.tolist())}")
    for col in ['Pemasukan','Pengeluaran','Saldo']:
        if col not in df.columns: continue
        df[col] = (df[col].astype(str)
            .str.replace(r'\(([0-9.,]+)\)', r'-\1', regex=True)
            .str.replace(r'[Rp\s$]|IDR', '', regex=True)
            .str.replace(r'\.(?=\d{3}(?:[.,]|$))', '', regex=True)
            .str.replace(',', '.', regex=False).str.strip())
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Tanggal']).sort_values('Tanggal').reset_index(drop=True)
    if df.empty: raise ValueError("Tidak ada baris valid. Periksa format kolom Tanggal.")
    df['Tahun'] = df['Tanggal'].dt.year; df['Bulan'] = df['Tanggal'].dt.month
    df['Hari']  = df['Tanggal'].dt.day;  df['BulanNama']   = df['Bulan'].map(MONTH_ID)
    df['BulanPendek'] = df['Bulan'].map(MONTH_SHORT)
    if 'Saldo' not in df.columns or df['Saldo'].abs().sum() == 0:
        df['Saldo'] = (df['Pemasukan'] - df['Pengeluaran']).cumsum()
    if 'Keterangan' not in df.columns: df['Keterangan'] = ''
    if 'Kategori'   not in df.columns: df['Kategori']   = 'Umum'
    return df

# ============================================================
# AGREGASI BULANAN
# ============================================================

def agregasi_bulanan(df: pd.DataFrame, tahun: int) -> pd.DataFrame:
    d = df[df['Tahun'] == tahun]
    if d.empty: return pd.DataFrame()
    m = (d.groupby(['Bulan','BulanNama','BulanPendek'])
         .agg(Pemasukan=('Pemasukan','sum'), Pengeluaran=('Pengeluaran','sum'),
              SaldoAkhir=('Saldo','last'), JmlTransaksi=('Pemasukan','count'))
         .reset_index().sort_values('Bulan'))
    m['Surplus']    = m['Pemasukan'] - m['Pengeluaran']
    m['RasioBeban'] = np.where(m['Pemasukan'] > 0, m['Pengeluaran'] / m['Pemasukan'] * 100, 100.0)
    m['Status']     = m['Surplus'].apply(lambda x: 'Surplus' if x >= 0 else 'Defisit')
    return m

# ============================================================
# DETEKSI ANOMALI
# ============================================================

def deteksi_anomali(monthly: pd.DataFrame) -> list:
    anomali = []
    if len(monthly) < 3: return anomali
    for col, label in [('Pengeluaran','Pengeluaran'),('Pemasukan','Pemasukan')]:
        mean_v = monthly[col].mean(); std_v = monthly[col].std()
        if std_v < 1: continue
        for _, row in monthly.iterrows():
            z = (row[col] - mean_v) / std_v
            if abs(z) > 2.0:
                arah = "sangat TINGGI" if z > 0 else "sangat RENDAH"
                anomali.append({'bulan': row['BulanNama'], 'kolom': label, 'nilai': row[col], 'arah': arah, 'z': z})
    return anomali

# ============================================================
# PREDIKSI
# ============================================================

def _exp_smoothing(values, alpha=0.45):
    if not values: return 0.0
    s = float(values[0])
    for v in values[1:]: s = alpha * float(v) + (1 - alpha) * s
    return s

def _holt_smoothing(values, alpha=0.5, beta=0.3, periods_ahead=3):
    n = len(values)
    if n < 2: return [values[-1]] * periods_ahead, list(values), [0.0] * n, list(values)
    L = [float(values[0])]; T = [float(values[1]) - float(values[0])]; fitted = [L[0] + T[0]]
    for i in range(1, n):
        v = float(values[i]); L_prev, T_prev = L[-1], T[-1]
        L_new = alpha * v + (1 - alpha) * (L_prev + T_prev)
        T_new = beta  * (L_new - L_prev) + (1 - beta) * T_prev
        L.append(L_new); T.append(T_new)
        if i < n - 1: fitted.append(L_new + T_new)
    forecasts = [max(0.0, L[-1] + h * T[-1]) for h in range(1, periods_ahead + 1)]
    return forecasts, L, T, fitted

def _mape(actual, predicted):
    a, p = np.array(actual, dtype=float), np.array(predicted, dtype=float)
    mask = a != 0
    if mask.sum() == 0: return 100.0
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)

def prediksi(monthly: pd.DataFrame, tahun: int) -> dict:
    n = len(monthly); avg_pem = float(monthly['Pemasukan'].mean()); avg_pen = float(monthly['Pengeluaran'].mean())
    last_sal = float(monthly['SaldoAkhir'].iloc[-1]); last_bln = int(monthly['Bulan'].iloc[-1])
    next_bln = (last_bln % 12) + 1; next_thn = tahun if last_bln < 12 else tahun + 1
    pem_vals = monthly['Pemasukan'].values.tolist(); pen_vals = monthly['Pengeluaran'].values.tolist()
    mae_pem = mae_pen = r2_pem = r2_pen = mape_pem = mape_pen = 0.0
    ci_pem_80 = ci_pen_80 = ci_pem_95 = ci_pen_95 = 0.0
    forecast_pem = forecast_pen = []; conf = "Rendah"; model_name = "—"

    if n < 2:
        pred_pem, pred_pen = avg_pem, avg_pen; forecast_pem = [avg_pem]*3; forecast_pen = [avg_pen]*3; model_name = "Rata-rata"
    elif n < 4:
        weights = np.arange(1, n+1, dtype=float)
        wma_pem = float(np.average(pem_vals, weights=weights)); wma_pen = float(np.average(pen_vals, weights=weights))
        ets_pem = _exp_smoothing(pem_vals); ets_pen = _exp_smoothing(pen_vals)
        pred_pem = max(0.0, 0.60*ets_pem + 0.40*wma_pem); pred_pen = max(0.0, 0.60*ets_pen + 0.40*wma_pen)
        forecast_pem = [pred_pem]*3; forecast_pen = [pred_pen]*3; model_name = "WMA + ETS"
    else:
        best_err = float('inf'); best_ap, best_bp, best_ae, best_be = 0.5, 0.3, 0.5, 0.3
        for ap in [0.2,0.35,0.5,0.65,0.8]:
            for bp in [0.1,0.2,0.3,0.4]:
                _, _, _, fitted_p = _holt_smoothing(pem_vals, ap, bp)
                if not fitted_p: continue
                ml = min(len(pem_vals)-1, len(fitted_p))
                err = np.mean(np.abs(np.array(pem_vals[1:ml+1]) - np.array(fitted_p[:ml])))
                if err < best_err: best_err = err; best_ap, best_bp = ap, bp
        best_err = float('inf')
        for ae in [0.2,0.35,0.5,0.65,0.8]:
            for be in [0.1,0.2,0.3,0.4]:
                _, _, _, fitted_e = _holt_smoothing(pen_vals, ae, be)
                if not fitted_e: continue
                ml = min(len(pen_vals)-1, len(fitted_e))
                err = np.mean(np.abs(np.array(pen_vals[1:ml+1]) - np.array(fitted_e[:ml])))
                if err < best_err: best_err = err; best_ae, best_be = ae, be
        fc_pem, Lp, Tp, fitted_p = _holt_smoothing(pem_vals, best_ap, best_bp, 3)
        fc_pen, Le, Te, fitted_e = _holt_smoothing(pen_vals, best_ae, best_be, 3)
        pred_pem = fc_pem[0]; pred_pen = fc_pen[0]; forecast_pem = fc_pem; forecast_pen = fc_pen
        res_p = np.array(pem_vals[1:]) - np.array([Lp[i]+Tp[i] for i in range(min(len(Lp),len(pem_vals))-1)])
        res_e = np.array(pen_vals[1:]) - np.array([Le[i]+Te[i] for i in range(min(len(Le),len(pen_vals))-1)])
        std_p = float(np.std(res_p)) if len(res_p) > 0 else avg_pem * 0.1
        std_e = float(np.std(res_e)) if len(res_e) > 0 else avg_pen * 0.1
        ci_pem_80 = 1.28*std_p; ci_pem_95 = 1.96*std_p; ci_pen_80 = 1.28*std_e; ci_pen_95 = 1.96*std_e
        xs = np.arange(1, n+1, dtype=float).reshape(-1,1); sp = max(2, int(n*0.75))
        if sp < n:
            m_p = LinearRegression().fit(xs[:sp], pem_vals[:sp]); m_e = LinearRegression().fit(xs[:sp], pen_vals[:sp])
            mae_pem = mean_absolute_error(pem_vals[sp:], m_p.predict(xs[sp:]))
            mae_pen = mean_absolute_error(pen_vals[sp:], m_e.predict(xs[sp:]))
        m_p_full = LinearRegression().fit(xs, pem_vals); m_e_full = LinearRegression().fit(xs, pen_vals)
        r2_pem = max(0.0, r2_score(pem_vals, m_p_full.predict(xs))); r2_pen = max(0.0, r2_score(pen_vals, m_e_full.predict(xs)))
        ml_p = min(len(pem_vals)-1, len(fitted_p)); ml_e = min(len(pen_vals)-1, len(fitted_e))
        if ml_p > 0: mape_pem = _mape(pem_vals[1:ml_p+1], fitted_p[:ml_p])
        if ml_e > 0: mape_pen = _mape(pen_vals[1:ml_e+1], fitted_e[:ml_e])
        avg_r2 = (r2_pem+r2_pen)/2; cv_pem = (monthly['Pemasukan'].std()/avg_pem*100) if avg_pem > 0 else 100
        avg_mape = (mape_pem+mape_pen)/2
        if avg_r2 >= 0.80 and cv_pem < 20 and avg_mape < 15: conf = "Tinggi"
        elif avg_r2 >= 0.50 or cv_pem < 40 or avg_mape < 30: conf = "Sedang"
        else: conf = "Rendah"
        model_name = f"Holt's DES (α={best_ap}, β={best_bp})"

    def next_month_name(base_bln, offset):
        bln = ((base_bln - 1 + offset) % 12) + 1
        thn = tahun + (last_bln - 1 + offset) // 12 - (last_bln - 1) // 12
        if last_bln == 12 and offset > 0: thn = tahun + (offset - 1) // 12 + 1
        return MONTH_ID[bln], bln

    fc3 = []
    for i in range(3):
        bn_nama, bn_num = next_month_name(last_bln, i+1)
        fp = forecast_pem[i] if i < len(forecast_pem) else pred_pem
        fe = forecast_pen[i] if i < len(forecast_pen) else pred_pen
        ci_p = ci_pem_80 * ((i+1)**0.5); ci_e = ci_pen_80 * ((i+1)**0.5)
        fc3.append({'bulan': bn_nama, 'bulan_num': bn_num, 'pemasukan': round(fp), 'pengeluaran': round(fe),
                    'surplus': round(fp - fe), 'ci_pem_80': round(ci_p), 'ci_pen_80': round(ci_e)})
    return {'pemasukan': round(pred_pem), 'pengeluaran': round(pred_pen),
            'surplus': round(pred_pem - pred_pen), 'saldo': round(last_sal + pred_pem - pred_pen),
            'bulan_nama': fc3[0]['bulan'] if fc3 else MONTH_ID[next_bln], 'tahun': next_thn,
            'confidence': conf, 'model': model_name, 'mae_pem': mae_pem, 'mae_pen': mae_pen,
            'mape_pem': mape_pem, 'mape_pen': mape_pen, 'r2_pem': r2_pem, 'r2_pen': r2_pen,
            'avg_pem': avg_pem, 'avg_pen': avg_pen, 'n_bulan': n,
            'ci_pem_80': ci_pem_80, 'ci_pem_95': ci_pem_95, 'ci_pen_80': ci_pen_80, 'ci_pen_95': ci_pen_95,
            'forecast_3bln': fc3}

# ============================================================
# HEALTH SCORE
# ============================================================

def hitung_health(monthly: pd.DataFrame) -> dict:
    ti, to = float(monthly['Pemasukan'].sum()), float(monthly['Pengeluaran'].sum()); n = len(monthly)
    tabungan   = max(0.0, min(100.0, (ti - to) / ti * 100)) if ti > 0 else 0.0
    stabilitas = float((monthly['Surplus'] >= 0).sum()) / n * 100 if n > 0 else 0.0
    if n >= 2:
        f = float(monthly['SaldoAkhir'].iloc[0]); l = float(monthly['SaldoAkhir'].iloc[-1])
        if f <= 0: ref = max(1.0, float(monthly['Pengeluaran'].mean())); pertumbuhan = max(0.0, min(100.0, (l/ref)*50))
        else: pct_grow = (l - f) / f * 100; pertumbuhan = max(0.0, min(100.0, pct_grow * 2))
    else: pertumbuhan = 50.0
    avg_pen = float(monthly['Pengeluaran'].mean()); last_sal = float(monthly['SaldoAkhir'].iloc[-1]) if n > 0 else 0.0
    if avg_pen > 0 and last_sal > 0: bln_aman = last_sal / avg_pen; cadangan = min(100.0, bln_aman / 3 * 100)
    elif last_sal <= 0: bln_aman, cadangan = 0.0, 0.0
    else: bln_aman, cadangan = 0.0, 50.0
    return {'tabungan': round(tabungan,1), 'stabilitas': round(stabilitas,1),
            'pertumbuhan': round(pertumbuhan,1), 'cadangan': round(cadangan,1), 'bln_aman': round(bln_aman,1)}

# ============================================================
# SIDEBAR
# ============================================================

@st.cache_data(show_spinner=False)
def _baca_cepat(file_bytes: bytes) -> pd.DataFrame:
    try:
        raw = pd.read_excel(io.BytesIO(file_bytes), header=0)
        if not any(k in ' '.join(raw.columns.str.lower()) for k in ['tanggal','date','tgl']):
            raw = pd.read_excel(io.BytesIO(file_bytes), header=1)
        return bersihkan_data(raw)
    except Exception: return pd.DataFrame()

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <div class="logo-wrap">
                <svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="3" y1="22" x2="21" y2="22"/>
                    <line x1="6" y1="18" x2="6" y2="11"/>
                    <line x1="10" y1="18" x2="10" y2="11"/>
                    <line x1="14" y1="18" x2="14" y2="11"/>
                    <line x1="18" y1="18" x2="18" y2="11"/>
                    <polygon points="12 2 20 7 4 7"/>
                </svg>
            </div>
            <span class="logo-title">Karang Taruna</span>
            <span class="logo-sub">Portal Keuangan Desa · v4.0</span>
        </div>
        
        <style>
        /* Membuat semua teks bawaan yang bertumpukan jadi transparan */
        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stFileUploaderDropzone"] section div {
            color: transparent !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            height: 0px !important;
            padding: 0px !important;
            margin: 0px !important;
        }
        /* Mengisi kotak dropzone dengan teks baru yang rapi */
        [data-testid="stFileUploaderDropzone"] section::before {
            content: "✨ Klik di sini untuk upload file";
            color: #E67E22;
            font-weight: 600;
            font-size: 0.95rem;
            display: block;
            text-align: center;
            padding: 20px 0;
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("### Unggah Data Keuangan")
        uploaded = st.file_uploader("", type=['xlsx','xls'], label_visibility="collapsed",
            help="Upload file rekap Excel (.xlsx/.xls). Kolom wajib: Tanggal, Pemasukan, Pengeluaran. Opsional: Saldo, Keterangan, Kategori")
        
        tahun_sel = None; bulan_sel = "Semua Bulan"

        if uploaded:
            df_preview = _baca_cepat(uploaded.getvalue())
            if not df_preview.empty:
                st.session_state['df_clean'] = df_preview
                st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
                st.markdown("### Filter Tampilan")
                tahun_list = sorted(df_preview['Tahun'].unique(), reverse=True)
                tahun_sel  = st.selectbox("Pilih Tahun", options=tahun_list, format_func=lambda x: f"Tahun {x}")
                bulan_tersedia = sorted(df_preview[df_preview['Tahun'] == tahun_sel]['Bulan'].unique())
                bulan_opts = ['Semua Bulan'] + [MONTH_ID[b] for b in bulan_tersedia]
                bulan_sel  = st.selectbox("Pilih Bulan", options=bulan_opts)

                df_fil = df_preview[df_preview['Tahun'] == tahun_sel]
                if bulan_sel != 'Semua Bulan':
                    bln_num = {v: k for k, v in MONTH_ID.items()}.get(bulan_sel)
                    if bln_num: df_fil = df_fil[df_fil['Bulan'] == bln_num]
                total_pem = df_fil['Pemasukan'].sum(); total_pen = df_fil['Pengeluaran'].sum()
                n_trx = len(df_fil); n_bln = df_fil['Bulan'].nunique()
                surplus = total_pem - total_pen
                label_periode = bulan_sel if bulan_sel != 'Semua Bulan' else f"Tahun {tahun_sel}"
                badge_cls = "surplus" if surplus >= 0 else "defisit"
                badge_txt = "Surplus" if surplus >= 0 else "Defisit"

                st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="sidebar-info">
                    <strong>{label_periode}</strong><br>
                    {n_trx} transaksi &nbsp;·&nbsp; {n_bln} bulan<br>
                    Masuk: <strong>{rupiah(total_pem, True)}</strong><br>
                    Keluar: <strong>{rupiah(total_pen, True)}</strong><br>
                    Selisih: <strong>{rupiah(surplus, True)}</strong><br><br>
                    <strong>Diperbarui</strong><br>
                    {datetime.now().strftime('%d %b %Y · %H:%M WIB')}
                    <br><span class="sidebar-badge {badge_cls}">{badge_txt}</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <hr class="sidebar-divider">
                <div class="sidebar-nav">
                    <strong>Menu Navigasi</strong><br>
                    <strong>Ringkasan</strong> — KPI &amp; Kesehatan<br>
                    <strong>Grafik</strong> — Visualisasi Data<br>
                    <strong>Prediksi</strong> — Perkiraan Bulan Depan<br>
                    <strong>Detail</strong> — Tabel Transaksi<br>
                    <strong>Tips</strong> — Saran Keuangan
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("File tidak valid. Pastikan ada kolom Tanggal, Pemasukan, Pengeluaran.")

        st.markdown("""
        <div class="sidebar-footer">
            Karang Taruna &nbsp;·&nbsp; Transparansi &amp; Kemajuan Bersama
        </div>""", unsafe_allow_html=True)
    return uploaded, tahun_sel, bulan_sel

# ============================================================
# WELCOME SCREEN
# ============================================================

def render_welcome():
    st.markdown("""
    <div class="welcome-box">
        <div class="welcome-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="3" y1="22" x2="21" y2="22"/>
                <line x1="6" y1="18" x2="6" y2="11"/>
                <line x1="10" y1="18" x2="10" y2="11"/>
                <line x1="14" y1="18" x2="14" y2="11"/>
                <line x1="18" y1="18" x2="18" y2="11"/>
                <polygon points="12 2 20 7 4 7"/>
            </svg>
        </div>
        <div class="welcome-title">Selamat Datang di Dashboard Keuangan</div>
        <div class="welcome-desc">
            Dashboard ini membantu warga desa dan anggota Karang Taruna memantau keuangan
            organisasi secara mudah, jelas, dan transparan — tanpa perlu memahami ilmu data.
        </div>
        <div class="welcome-steps">
            <div class="step-item">
                <div class="step-num">1</div>
                <div class="step-title">Unggah Data Excel</div>
                <div class="step-desc">Klik tombol upload di sidebar kiri, lalu pilih file Excel keuangan Anda.</div>
            </div>
            <div class="step-item">
                <div class="step-num">2</div>
                <div class="step-title">Pilih Tahun / Bulan</div>
                <div class="step-desc">Pilih tahun atau bulan yang ingin Anda lihat lewat filter di sidebar.</div>
            </div>
            <div class="step-item">
                <div class="step-num">3</div>
                <div class="step-title">Lihat Hasilnya</div>
                <div class="step-desc">Grafik, prediksi 3 bulan, dan saran keuangan tampil otomatis.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-title">Format File Excel yang Dibutuhkan</div>', unsafe_allow_html=True)
    contoh = pd.DataFrame({'Tanggal':['01/01/2024','07/01/2024','14/01/2024','21/01/2024'],
        'Pemasukan':['500.000','750.000','1.200.000','300.000'], 'Pengeluaran':['200.000','400.000','600.000','150.000'],
        'Saldo':['300.000','650.000','1.250.000','1.400.000'],
        'Keterangan':['Iuran Warga','Donasi','Dana Desa','Arisan'], 'Kategori':['Iuran','Donasi','Dana Pemerintah','Sosial']})
    st.dataframe(contoh, use_container_width=True, hide_index=True)
    st.info("Kolom **Tanggal, Pemasukan, Pengeluaran** wajib ada. Kolom **Saldo, Keterangan, Kategori** opsional. Angka boleh pakai titik ribuan (1.000.000) atau tanpa titik.")

# ============================================================
# KPI CARDS
# ============================================================

def _delta_label(monthly, col):
    if len(monthly) < 2: return ""
    last = monthly[col].iloc[-1]; prev = monthly[col].iloc[-2]
    if prev == 0: return ""
    d = (last - prev) / prev * 100; arrow = "▲" if d >= 0 else "▼"
    color = CLR_GREEN if d >= 0 else CLR_RED
    return f'<div class="kpi-delta" style="color:{color};">{arrow} {abs(d):.1f}% vs bulan lalu</div>'

def render_kpi(monthly, pred):
    ti = monthly['Pemasukan'].sum(); to = monthly['Pengeluaran'].sum()
    sur = ti - to; sal = monthly['SaldoAkhir'].iloc[-1]; pct = (sur / ti * 100) if ti > 0 else 0.0
    kartu = [
        {'w': CLR_NAVY,  'icon':'🏦', 'label':'Saldo Kas',        'nilai': rupiah(sal, True), 'sub':'Uang tunai tersedia saat ini', 'delta':''},
        {'w': CLR_GREEN, 'icon':'↑',  'label':'Total Pemasukan',  'nilai': rupiah(ti, True),  'sub':'Semua uang yang masuk',        'delta': _delta_label(monthly,'Pemasukan')},
        {'w': CLR_RED,   'icon':'↓',  'label':'Total Pengeluaran','nilai': rupiah(to, True),  'sub':'Semua uang yang keluar',       'delta': _delta_label(monthly,'Pengeluaran')},
        {'w': CLR_GREEN if sur >= 0 else CLR_RED,
         'icon': '▲' if sur >= 0 else '▼',
         'label':'Surplus / Defisit', 'nilai': rupiah(sur, True),
         'sub':'Keuangan sehat' if sur >= 0 else 'Pengeluaran melebihi pemasukan', 'delta':''},
        {'w': CLR_GOLD, 'icon':'%', 'label':'Persentase Ditabung', 'nilai': f"{max(0, pct):.1f}%",
         'sub': f"dari pemasukan  ·  {'Target >20% tercapai' if pct >= 20 else 'Target >20%'}", 'delta':''},
    ]
    cols = st.columns(5)
    for col, k in zip(cols, kartu):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color:{k['w']};">
                <div style="font-size:1.4rem;margin-bottom:6px;color:{k['w']};font-weight:800;line-height:1;">{k['icon']}</div>
                <div class="kpi-label">{k['label']}</div>
                <div class="kpi-value">{k['nilai']}</div>
                <div class="kpi-sub">{k['sub']}</div>
                {k['delta']}
            </div>""", unsafe_allow_html=True)

# ============================================================
# PREDIKSI BOX + FORECAST 3 BULAN
# ============================================================

def render_prediksi(pred, monthly):
    sur = pred['surplus']; t = "+" if sur >= 0 else ""
    dp = ((pred['pemasukan'] - pred['avg_pem']) / pred['avg_pem'] * 100) if pred['avg_pem'] > 0 else 0
    de = ((pred['pengeluaran'] - pred['avg_pen']) / pred['avg_pen'] * 100) if pred['avg_pen'] > 0 else 0
    st.markdown(f"""
    <div class="pred-box">
        <div class="pred-box-title">Prediksi Keuangan — {pred['bulan_nama']} {pred['tahun']}</div>
        <div class="pred-box-sub">
            Model: <strong>{pred['model']}</strong> &nbsp;·&nbsp;
            Data: {pred['n_bulan']} bulan &nbsp;·&nbsp;
            Kepercayaan: <strong>{pred['confidence']}</strong>
        </div>
        <div class="pred-grid">
            <div class="pred-item">
                <div class="pred-item-label">Perkiraan Pemasukan</div>
                <div class="pred-item-value">{rupiah(pred['pemasukan'],True)}</div>
                <div class="pred-item-delta">{"▲" if dp>=0 else "▼"} {abs(dp):.1f}% vs rata-rata</div>
            </div>
            <div class="pred-item">
                <div class="pred-item-label">Perkiraan Pengeluaran</div>
                <div class="pred-item-value">{rupiah(pred['pengeluaran'],True)}</div>
                <div class="pred-item-delta">{"▲" if de>=0 else "▼"} {abs(de):.1f}% vs rata-rata</div>
            </div>
            <div class="pred-item">
                <div class="pred-item-label">Perkiraan {"Surplus" if sur>=0 else "Defisit"}</div>
                <div class="pred-item-value">{t}{rupiah(sur,True)}</div>
                <div class="pred-item-delta">{"Keuangan aman" if sur>=0 else "Siapkan cadangan"}</div>
            </div>
            <div class="pred-item">
                <div class="pred-item-label">Prediksi Saldo Akhir</div>
                <div class="pred-item-value">{rupiah(pred['saldo'],True)}</div>
                <div class="pred-item-delta">Setelah {pred['bulan_nama']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if pred['n_bulan'] < 3:
        st.warning(f"Data hanya {pred['n_bulan']} bulan — prediksi masih kasar. Semakin banyak data, prediksi semakin akurat.")

    st.markdown('<div class="section-title">Perkiraan 3 Bulan ke Depan</div>', unsafe_allow_html=True)
    fc3 = pred.get('forecast_3bln', [])
    if fc3:
        cols = st.columns(3)
        for i, fc in enumerate(fc3):
            with cols[i]:
                sur_fc = fc['surplus']; clr = CLR_NAVY if sur_fc >= 0 else CLR_RED; ci_p = fc.get('ci_pem_80', 0)
                st.markdown(f"""
                <div class="kpi-card" style="border-top-color:{clr}; padding-top:18px;">
                    <div class="kpi-label">Bulan {i+1} — {fc['bulan']}</div>
                    <div class="kpi-value" style="color:{clr};">{'+'if sur_fc>=0 else ''}{rupiah(sur_fc, True)}</div>
                    <div class="kpi-sub">
                        Masuk: {rupiah(fc['pemasukan'],True)}<br>
                        Keluar: {rupiah(fc['pengeluaran'],True)}
                    </div>
                    <div class="kpi-delta" style="color:#8090B5; font-size:0.60rem; margin-top:5px;">
                        ± {rupiah(ci_p, True)} (rentang 80%)
                    </div>
                </div>""", unsafe_allow_html=True)

    if len(monthly) >= 3:
        st.plotly_chart(chart_forecast(monthly, pred), use_container_width=True)

    sur_clr = CLR_GREEN if sur >= 0 else CLR_RED
    pesan_sur = "Lebih banyak uang masuk dari keluar — keuangan aman." if sur >= 0 else "Pengeluaran lebih besar dari pemasukan — siapkan tabungan cadangan."
    st.markdown(f"""
    <div style="background:#EEF3FB;border-radius:14px;padding:20px 22px;border-left:5px solid var(--navy);margin-top:6px;font-size:0.85rem;color:#1A2D5A;line-height:1.80;">
        <div style="font-weight:800;font-size:0.95rem;color:#1B3A6B;margin-bottom:12px;">Ringkasan Prediksi — {pred['bulan_nama']} {pred['tahun']}</div>
        Perkiraan <strong>Pemasukan</strong>: <strong>{rupiah(pred['pemasukan'])}</strong><br>
        Perkiraan <strong>Pengeluaran</strong>: <strong>{rupiah(pred['pengeluaran'])}</strong><br>
        <span style="color:{sur_clr};font-weight:700;">{pesan_sur}</span><br>
        Prediksi <strong>Saldo Akhir</strong>: <strong>{rupiah(pred['saldo'])}</strong><br>
        Kepercayaan Model: <strong>{pred['confidence']}</strong>
        <div style="margin-top:14px;padding-top:10px;border-top:1px dashed #C9D6EE;font-size:0.76rem;color:#6B7A99;">
            Ini perkiraan berbasis tren historis. Tetap catat semua transaksi agar prediksi semakin akurat.
        </div>
    </div>
    """, unsafe_allow_html=True)

    anomali = deteksi_anomali(monthly)
    if anomali:
        st.markdown('<div class="section-title">Anomali Keuangan Terdeteksi</div>', unsafe_allow_html=True)
        for a in anomali:
            badge = "Naik" if "TINGGI" in a['arah'] else "Turun"
            st.markdown(f"""
            <div class="anomali-box">
                <div class="anomali-badge">{badge}</div>
                <div>
                    <div class="anomali-title">Bulan {a['bulan']} — {a['kolom']} {a['arah']}</div>
                    <div class="anomali-text">Nilai tercatat: <strong>{rupiah(a['nilai'], True)}</strong> — sangat jauh dari rata-rata (z-score = {a['z']:.1f}). Periksa kembali transaksi bulan ini.</div>
                </div>
            </div>""", unsafe_allow_html=True)

# ============================================================
# GRAFIK
# ============================================================

def chart_saldo(monthly):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly['BulanNama'], y=monthly['SaldoAkhir'], mode='lines+markers', name='Saldo Aktual',
        line=dict(color=CLR_NAVY, width=3), fill='tozeroy', fillcolor='rgba(27,58,107,0.10)',
        marker=dict(size=9, color=CLR_NAVY), hovertemplate='<b>%{x}</b><br>Saldo: Rp %{y:,.0f}<extra></extra>'))
    if len(monthly) >= 2:
        xs = np.arange(len(monthly), dtype=float).reshape(-1,1)
        trend = LinearRegression().fit(xs, monthly['SaldoAkhir'].values).predict(xs)
        fig.add_trace(go.Scatter(x=monthly['BulanNama'], y=trend, mode='lines', name='Garis Tren',
            line=dict(color=CLR_GOLD, width=2.5, dash='dot'), hovertemplate='<b>%{x}</b><br>Tren: Rp %{y:,.0f}<extra></extra>'))
    fig.update_layout(title=dict(text="Pergerakan Saldo Kas + Garis Tren", font=dict(size=13, color=CLR_DARK)),
        yaxis=dict(tickformat=",", tickprefix="Rp ", gridcolor="#DDE3EF"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"), legend=dict(orientation="h", y=1.10, x=0), **LAYOUT)
    return fig

def chart_bulanan(monthly):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly['BulanPendek'], y=monthly['Pemasukan'], name='Pemasukan',
        marker_color=CLR_GREEN, marker_line_width=0, hovertemplate='%{x}<br>Pemasukan: Rp %{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Bar(x=monthly['BulanPendek'], y=monthly['Pengeluaran'], name='Pengeluaran',
        marker_color=CLR_RED, marker_line_width=0, hovertemplate='%{x}<br>Pengeluaran: Rp %{y:,.0f}<extra></extra>'))
    fig.update_layout(title=dict(text="Pemasukan vs Pengeluaran per Bulan", font=dict(size=13, color=CLR_DARK)),
        barmode='group', yaxis=dict(tickformat=",", gridcolor="#DDE3EF"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"), legend=dict(orientation="h", y=1.10, x=0), **LAYOUT)
    return fig

def chart_surplus(monthly):
    colors = [CLR_GREEN if s >= 0 else CLR_RED for s in monthly['Surplus']]
    fig = go.Figure(go.Bar(x=monthly['BulanPendek'], y=monthly['Surplus'],
        marker_color=colors, marker_line_width=0, hovertemplate='%{x}<br>Rp %{y:,.0f}<extra></extra>'))
    fig.add_hline(y=0, line_dash="dash", line_color="#1B3A6B", line_width=1.5,
                  annotation_text="Titik Impas", annotation_position="top right", annotation_font_size=10)
    fig.update_layout(title=dict(text="Surplus / Defisit Tiap Bulan", font=dict(size=13, color=CLR_DARK)),
        yaxis=dict(tickformat=",", gridcolor="#DDE3EF"), xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False, **LAYOUT)
    return fig

def chart_donut(ti, to):
    if ti <= 0 and to <= 0: return go.Figure()
    if ti >= to:
        sur = ti - to; labels = ['Pengeluaran', 'Surplus / Ditabung']; values = [to, sur]; colors = [CLR_RED, CLR_GREEN]
        pct = sur / ti * 100 if ti > 0 else 0
        center = f"<b>{pct:.1f}%</b><br><span style='font-size:10px'>ditabung</span>"
    else:
        kelebihan = to - ti; labels = ['Ditanggung Pemasukan', 'Kelebihan Pengeluaran']; values = [ti, kelebihan]; colors = [CLR_BLUE, CLR_RED]
        pct = kelebihan / to * 100 if to > 0 else 0
        center = f"<b style='color:#E74C3C;'>defisit</b><br><span style='font-size:10px'>{pct:.1f}%</span>"
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.62,
        marker=dict(colors=colors, line=dict(color='white', width=3)),
        hovertemplate='<b>%{label}</b><br>Rp %{value:,.0f} (%{percent})<extra></extra>'))
    fig.add_annotation(text=center, x=0.5, y=0.5, showarrow=False, font=dict(size=16, color=CLR_DARK))
    fig.update_layout(title=dict(text="Komposisi Penggunaan Dana", font=dict(size=13, color=CLR_DARK)),
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Poppins, sans-serif", size=11), margin=dict(l=12,r=12,t=50,b=16))
    return fig

def chart_forecast(monthly, pred):
    fc3 = pred.get('forecast_3bln', [])
    if not fc3: return go.Figure()
    hist_x = monthly['BulanNama'].tolist(); hist_p = monthly['Pemasukan'].tolist(); hist_e = monthly['Pengeluaran'].tolist()
    fc_x = [f['bulan'] for f in fc3]; fc_p = [f['pemasukan'] for f in fc3]; fc_e = [f['pengeluaran'] for f in fc3]
    ci_p_hi = [f['pemasukan']+f.get('ci_pem_80',0)*((i+1)**0.5) for i,f in enumerate(fc3)]
    ci_p_lo = [max(0,f['pemasukan']-f.get('ci_pem_80',0)*((i+1)**0.5)) for i,f in enumerate(fc3)]
    ci_e_hi = [f['pengeluaran']+f.get('ci_pen_80',0)*((i+1)**0.5) for i,f in enumerate(fc3)]
    ci_e_lo = [max(0,f['pengeluaran']-f.get('ci_pen_80',0)*((i+1)**0.5)) for i,f in enumerate(fc3)]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Forecast Pemasukan","Forecast Pengeluaran"), horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=hist_x, y=hist_p, name='Aktual Pemasukan', line=dict(color=CLR_GREEN, width=2.5), marker=dict(size=7), mode='lines+markers', hovertemplate='%{x}: Rp %{y:,.0f}<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(x=fc_x, y=ci_p_hi, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=fc_x, y=ci_p_lo, name='Rentang Prediksi (80%)', mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(39,174,96,0.18)', showlegend=True, hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=fc_x, y=fc_p, name='Forecast Pemasukan', mode='lines+markers', line=dict(color=CLR_GREEN, width=2, dash='dot'), marker=dict(size=9, symbol='diamond', color=CLR_GREEN), hovertemplate='%{x}: Rp %{y:,.0f}<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist_x, y=hist_e, name='Aktual Pengeluaran', line=dict(color=CLR_RED, width=2.5), marker=dict(size=7), mode='lines+markers', hovertemplate='%{x}: Rp %{y:,.0f}<extra></extra>'), row=1, col=2)
    fig.add_trace(go.Scatter(x=fc_x, y=ci_e_hi, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=2)
    fig.add_trace(go.Scatter(x=fc_x, y=ci_e_lo, name='Rentang Prediksi Pengeluaran', mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(231,76,60,0.18)', showlegend=True, hoverinfo='skip'), row=1, col=2)
    fig.add_trace(go.Scatter(x=fc_x, y=fc_e, name='Forecast Pengeluaran', mode='lines+markers', line=dict(color=CLR_RED, width=2, dash='dot'), marker=dict(size=9, symbol='diamond', color=CLR_RED), hovertemplate='%{x}: Rp %{y:,.0f}<extra></extra>'), row=1, col=2)
    fig.update_layout(title=dict(text=f"Visualisasi Perkiraan 3 Bulan ke Depan ({pred['model']})", font=dict(size=13, color=CLR_DARK)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Poppins, sans-serif", size=11, color="#1A2D5A"),
        margin=dict(l=12,r=12,t=80,b=80), legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center", bgcolor="rgba(255,255,255,0.90)", bordercolor="#DDE3EF", borderwidth=1, font=dict(size=10)))
    for ann in fig.layout.annotations: ann.update(y=ann.y-0.04, font=dict(size=11, color="#1A2D5A"))
    fig.update_yaxes(tickformat=",", tickprefix="Rp ", gridcolor="#DDE3EF"); fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    return fig

def chart_akurasi(monthly):
    n = len(monthly)
    if n < 4: return None
    xs = np.arange(1, n+1, dtype=float).reshape(-1,1); sp = max(2, int(n*0.75))
    if sp >= n: return None
    mp = LinearRegression().fit(xs[:sp], monthly['Pemasukan'].values[:sp])
    me = LinearRegression().fit(xs[:sp], monthly['Pengeluaran'].values[:sp])
    pp = mp.predict(xs[sp:]); pe = me.predict(xs[sp:]); xl = monthly['BulanNama'].values[sp:]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Akurasi Prediksi Pemasukan","Akurasi Prediksi Pengeluaran"))
    for col_idx, (actuals, preds, clr, name) in enumerate([(monthly['Pemasukan'].values[sp:], pp, CLR_GREEN, 'Pemasukan'),(monthly['Pengeluaran'].values[sp:], pe, CLR_RED, 'Pengeluaran')], 1):
        fig.add_trace(go.Scatter(x=xl, y=actuals, name=f'Aktual {name}', line=dict(color=clr, width=2.5), marker=dict(size=8), mode='lines+markers', showlegend=(col_idx==1)), row=1, col=col_idx)
        fig.add_trace(go.Scatter(x=xl, y=preds, name=f'Prediksi {name}', line=dict(color=clr, width=2.5, dash='dot'), marker=dict(size=8, symbol='diamond'), showlegend=(col_idx==1)), row=1, col=col_idx)
    fig.update_layout(title=dict(text="Akurasi Prediksi: Data Nyata vs Perkiraan", font=dict(size=13, color=CLR_DARK)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Poppins, sans-serif", size=11, color="#1A2D5A"),
        margin=dict(l=12,r=12,t=80,b=80), legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center", bgcolor="rgba(255,255,255,0.90)", bordercolor="#DDE3EF", borderwidth=1, font=dict(size=10)))
    for ann in fig.layout.annotations: ann.update(y=ann.y-0.04, font=dict(size=11, color="#1A2D5A"))
    fig.update_yaxes(tickformat=",", tickprefix="Rp ", gridcolor="#DDE3EF"); fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    return fig

def chart_kategori(df, tahun):
    d = df[df['Tahun'] == tahun]
    if 'Kategori' not in d.columns: return None
    unique_kat = d['Kategori'].dropna().unique()
    if len(unique_kat) <= 1 and (len(unique_kat) == 0 or unique_kat[0] == 'Umum'): return None
    kat = (d.groupby('Kategori').agg(Pengeluaran=('Pengeluaran','sum'), Pemasukan=('Pemasukan','sum')).reset_index().sort_values('Pengeluaran', ascending=True))
    fig = go.Figure()
    fig.add_trace(go.Bar(y=kat['Kategori'], x=kat['Pengeluaran'], orientation='h', name='Pengeluaran', marker_color=CLR_RED, marker_line_width=0, hovertemplate='%{y}<br>Pengeluaran: Rp %{x:,.0f}<extra></extra>'))
    fig.add_trace(go.Bar(y=kat['Kategori'], x=kat['Pemasukan'], orientation='h', name='Pemasukan', marker_color=CLR_GREEN, marker_line_width=0, hovertemplate='%{y}<br>Pemasukan: Rp %{x:,.0f}<extra></extra>'))
    fig.update_layout(title=dict(text="Pemasukan & Pengeluaran per Kategori", font=dict(size=13, color=CLR_DARK)),
        barmode='group', xaxis=dict(tickformat=",", gridcolor="#DDE3EF"), yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", y=1.10, x=0), **LAYOUT)
    return fig

def chart_yoy(df, tahun_list):
    records = []
    for th in tahun_list:
        m = agregasi_bulanan(df, th)
        if m.empty: continue
        for _, row in m.iterrows():
            records.append({'Tahun': str(th), 'Bulan': row['Bulan'], 'BulanPendek': row['BulanPendek'], 'Pemasukan': row['Pemasukan'], 'Pengeluaran': row['Pengeluaran']})
    if not records: return None
    df_yoy = pd.DataFrame(records); colors_list = [CLR_NAVY, CLR_GREEN, CLR_GOLD, "#E67E22"]
    fig = go.Figure()
    for i, th in enumerate(tahun_list):
        sub = df_yoy[df_yoy['Tahun']==str(th)].sort_values('Bulan'); clr = colors_list[i % len(colors_list)]
        fig.add_trace(go.Scatter(x=sub['BulanPendek'], y=sub['Pemasukan'], mode='lines+markers', name=f'Pemasukan {th}',
            line=dict(color=clr, width=2), marker=dict(size=7), hovertemplate=f'%{{x}} {th}: Rp %{{y:,.0f}}<extra></extra>'))
    fig.update_layout(title=dict(text="Perbandingan Pemasukan Tahun ke Tahun", font=dict(size=13, color=CLR_DARK)),
        yaxis=dict(tickformat=",", tickprefix="Rp ", gridcolor="#DDE3EF"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"), legend=dict(orientation="h", y=1.10, x=0), **LAYOUT)
    return fig

# ============================================================
# HEALTH BARS
# ============================================================

def render_health(monthly):
    h = hitung_health(monthly)
    bars = [
        {'label':'Tingkat Tabungan',            'val': h['tabungan'],    'sub': f"{h['tabungan']:.1f}% dari pemasukan berhasil ditabung  ·  Target sehat: di atas 20%"},
        {'label':'Bulan dengan Keuangan Positif','val': h['stabilitas'],  'sub': f"{h['stabilitas']:.0f}% bulan berjalan surplus  ·  Semakin tinggi semakin stabil"},
        {'label':'Pertumbuhan Saldo',             'val': h['pertumbuhan'],'sub': f"Skor pertumbuhan: {h['pertumbuhan']:.1f}%  ·  Menandakan kas makin bertambah"},
        {'label':'Cadangan Kas Darurat',          'val': h['cadangan'],   'sub': f"Cukup untuk ≈ {h['bln_aman']} bulan ke depan  ·  Idealnya 2–3 bulan"},
    ]
    col_bars, col_score = st.columns([3, 1])
    with col_bars:
        for b in bars:
            c = clr_health(b['val'])
            st.markdown(f"""
            <div class="health-row">
                <div class="health-meta">
                    <span class="health-label">{b['label']}</span>
                    <span class="health-score" style="color:{c};">{dot_health(b['val'])} {b['val']:.0f}%</span>
                </div>
                <div class="health-bar-bg">
                    <div class="health-bar-fill" style="width:{b['val']}%; background:{c};"></div>
                </div>
                <div class="health-sub">{b['sub']}</div>
            </div>""", unsafe_allow_html=True)
    with col_score:
        avg = float(np.mean([h['tabungan'],h['stabilitas'],h['pertumbuhan'],h['cadangan']]))
        c = clr_health(avg)
        sts = "Sangat Sehat" if avg >= 70 else ("Perlu Perhatian" if avg >= 40 else "Kritis")
        st.markdown(f"""
        <div class="score-circle" style="background:{c}18; border:3px solid {c};">
            <div class="score-num" style="color:{c};">{avg:.0f}%</div>
            <div class="score-label" style="color:{c};">{sts}</div>
            <div class="score-sub">Skor Kesehatan<br>Keuangan Keseluruhan</div>
        </div>""", unsafe_allow_html=True)

# ============================================================
# A/B TEST
# ============================================================

def render_ab(df, tahun):
    dy = df[df['Tahun'] == tahun]
    a  = dy[dy['Hari'] <= 15]['Pemasukan'].dropna(); k = dy[dy['Hari'] > 15]['Pemasukan'].dropna()
    aa = float(a.mean()) if len(a) > 0 else 0.0; ak = float(k.mean()) if len(k) > 0 else 0.0
    aa = 0.0 if np.isnan(aa) else aa; ak = 0.0 if np.isnan(ak) else ak
    denom = max(1.0, (aa + ak) / 2); pct = abs(aa - ak) / denom * 100
    sig, pv = False, 1.0
    if len(a) >= 2 and len(k) >= 2: _, pv = ttest_ind(a, k, equal_var=False); sig = bool(pv < 0.05)
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        st.markdown(f"""
        <div class="ab-card awal">
            <div class="ab-period">Awal Bulan (Tgl 1–15)</div>
            <div class="ab-value">{rupiah(aa, True)}</div>
            <div class="ab-desc">Rata-rata pemasukan per transaksi</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="ab-card akhir">
            <div class="ab-period">Akhir Bulan (Tgl 16–31)</div>
            <div class="ab-value">{rupiah(ak, True)}</div>
            <div class="ab-desc">Rata-rata pemasukan per transaksi</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        if len(a) < 2 or len(k) < 2:
            st.markdown(f"""
            <div style="background-color: #F0F4F8; color: #4A5778; padding: 16px; border-radius: 12px; border: 2px solid #DDE3EF; font-size: 0.85rem; line-height: 1.5;">
                Data belum cukup untuk uji statistik (butuh minimal 2 transaksi per periode). Selisih saat ini: <strong>{pct:.1f}%</strong>.
            </div>
            """, unsafe_allow_html=True)
        elif sig:
            lb = "awal bulan" if aa > ak else "akhir bulan"
            st.markdown(f"""
            <div class="ab-card" style="background: var(--orange-light); border: 2px solid var(--orange); height: 100%; box-sizing: border-box;">
                <div class="ab-period">Hasil Analisis Pola</div>
                <div class="ab-value" style="color: var(--orange-dark); font-size: 1.38rem;">{pct:.1f}%</div>
                <div class="ab-desc">Signifikan di {lb}. Rencanakan pengeluaran saat kas penuh.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="ab-card" style="background: #E8F8EF; border: 2px solid #A9DFBF; height: 100%; box-sizing: border-box;">
                <div class="ab-period">Hasil Analisis Pola</div>
                <div class="ab-value" style="color: var(--green); font-size: 1.38rem;">{pct:.1f}%</div>
                <div class="ab-desc">Pemasukan merata sepanjang bulan. Arus kas Anda cenderung stabil.</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# TIPS & SARAN
# ============================================================

def render_tips(monthly, pred):
    ti = monthly['Pemasukan'].sum(); to = monthly['Pengeluaran'].sum()
    r = to / ti if ti > 0 else 1.0; sur = pred['surplus']; tips = []
    if r > 0.85:
        tips.append({'badge':'!','w':CLR_RED,'judul':'Pengeluaran Terlalu Besar',
            'isi':f'Lebih dari 85% pemasukan habis untuk pengeluaran ({r*100:.0f}%). Coba kurangi belanja yang tidak mendesak, atau cari sumber pemasukan tambahan.'})
    elif r > 0.70:
        tips.append({'badge':'i','w':CLR_GOLD,'judul':'Pengeluaran Perlu Diperhatikan',
            'isi':f'Pengeluaran sudah {r*100:.0f}% dari pemasukan. Masih aman, tapi mulailah menabung lebih banyak agar ada cadangan dana mendadak.'})
    else:
        tips.append({'badge':'OK','w':CLR_GREEN,'judul':'Keuangan Sangat Sehat',
            'isi':f'Pengeluaran hanya {r*100:.0f}% dari pemasukan. Pertahankan pola ini dan pertimbangkan menyisihkan sebagian untuk kegiatan produktif desa.'})
    if sur < 0:
        tips.append({'badge':'!','w':CLR_RED,'judul':f'Waspadai Bulan {pred["bulan_nama"]}',
            'isi':f'Prediksi bulan depan: pengeluaran lebih besar dari pemasukan (defisit ≈ {rupiah(abs(sur),True)}). Siapkan dana cadangan dari sekarang.'})
    else:
        tips.append({'badge':'OK','w':CLR_GREEN,'judul':f'Proyeksi Bulan {pred["bulan_nama"]} Positif',
            'isi':f'Perkiraan surplus ≈ {rupiah(sur,True)} bulan depan. Manfaatkan untuk menambah tabungan atau kegiatan sosial bagi warga desa.'})
    best = monthly.loc[monthly['Pemasukan'].idxmax()]
    tips.append({'badge':'#1','w':CLR_NAVY,'judul':f'Bulan Terkuat: {best["BulanNama"]}',
        'isi':f'Pemasukan tertinggi di bulan {best["BulanNama"]} ({rupiah(best["Pemasukan"],True)}). Jadwalkan kegiatan atau pembelian penting saat kas sedang penuh.'})
    tips.append({'badge':'N','w':CLR_GREEN,'judul':'Selalu Catat Setiap Transaksi',
        'isi':'Catat semua pemasukan dan pengeluaran tepat waktu dengan tanggal yang benar. Rekap mingguan membantu agar tidak ada yang terlewat dan laporan selalu akurat.'})
    cols = st.columns(2)
    for i, t in enumerate(tips):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="tip-card" style="border-color:{t['w']};">
                <div class="tip-badge" style="background:{t['w']};">{t['badge']}</div>
                <div>
                    <div class="tip-title">{t['judul']}</div>
                    <div class="tip-text">{t['isi']}</div>
                </div>
            </div>""", unsafe_allow_html=True)

# ============================================================
# TABEL REKAP
# ============================================================

def render_tabel(monthly, tahun):
    tampil = monthly[['BulanNama','Pemasukan','Pengeluaran','Surplus','RasioBeban','SaldoAkhir','JmlTransaksi','Status']].copy()
    tampil.columns = ['Bulan','Pemasukan','Pengeluaran','Surplus/Defisit','Rasio Beban','Saldo Akhir','Jml. Transaksi','Status']
    for col in ['Pemasukan','Pengeluaran','Surplus/Defisit','Saldo Akhir']: tampil[col] = tampil[col].apply(rupiah)
    tampil['Rasio Beban'] = tampil['Rasio Beban'].apply(lambda x: f"{x:.1f}%")
    st.dataframe(tampil, use_container_width=True, hide_index=True)
    buf = io.BytesIO(); monthly.to_excel(buf, index=False, engine='openpyxl'); buf.seek(0)
    st.download_button(label="Download Rekap Excel", data=buf.getvalue(),
        file_name=f"Rekap_Keuangan_Karang_Taruna_{tahun}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ============================================================
# AKURASI MODEL
# ============================================================

def render_akurasi(monthly, pred):
    if len(monthly) < 4:
        st.info("Butuh minimal 4 bulan data untuk menampilkan grafik validasi prediksi.")
        return
    fig_ak = chart_akurasi(monthly)
    if fig_ak: st.plotly_chart(fig_ak, use_container_width=True)
    conf = pred.get('confidence', 'Rendah'); model = pred.get('model', '—'); n_bulan = pred.get('n_bulan', 0)
    if 'Tinggi' in conf: clr_conf = "#27AE60"; pesan_conf = "Prediksi sangat bisa diandalkan."
    elif 'Sedang' in conf: clr_conf = "#F39C12"; pesan_conf = "Prediksi cukup bisa diandalkan, namun bisa sedikit meleset."
    else: clr_conf = "#E74C3C"; pesan_conf = "Prediksi masih kasar karena data belum cukup banyak."
    st.markdown(f"""
    <div style="background:white;border-radius:16px;padding:22px 24px;box-shadow:0 3px 14px rgba(27,58,107,0.09);border-left:5px solid #1B3A6B;margin-top:10px;">
        <div style="font-size:0.94rem;font-weight:800;color:#1B3A6B;margin-bottom:16px;">Seberapa Akurat Prediksi Ini?</div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;">
            <div style="background:#EEF3FB;border-radius:12px;padding:16px;text-align:center;">
                <div style="width:14px;height:14px;border-radius:50%;background:{clr_conf};margin:0 auto 10px;"></div>
                <div style="font-size:0.72rem;font-weight:700;color:#8090B5;margin-bottom:4px;">Tingkat Kepercayaan</div>
                <div style="font-size:0.92rem;font-weight:900;color:#1B3A6B;">{conf}</div>
            </div>
            <div style="background:#EEF3FB;border-radius:12px;padding:16px;text-align:center;">
                <div style="font-size:1.3rem;font-weight:800;color:#1B3A6B;margin-bottom:6px;">{n_bulan}</div>
                <div style="font-size:0.72rem;font-weight:700;color:#8090B5;margin-bottom:4px;">Jumlah Data</div>
                <div style="font-size:0.92rem;font-weight:900;color:#1B3A6B;">bulan</div>
            </div>
            <div style="background:#EEF3FB;border-radius:12px;padding:16px;text-align:center;">
                <div style="font-size:0.65rem;font-weight:800;color:#1B3A6B;margin-bottom:6px;word-break:break-all;">{model}</div>
                <div style="font-size:0.72rem;font-weight:700;color:#8090B5;margin-bottom:4px;">Metode Prediksi</div>
            </div>
        </div>
        <div style="margin-top:16px;background:#EEF3FB;border-radius:12px;padding:14px 18px;font-size:0.82rem;color:#1A2D5A;line-height:1.75;">
            <strong>Cara membaca grafik:</strong> Garis <em>penuh</em> adalah data nyata, garis <em>putus-putus</em> adalah prediksi. Semakin berdekatan kedua garis, semakin akurat modelnya.<br><br>
            <span style="color:{clr_conf};font-weight:700;">{pesan_conf}</span> Semakin banyak data yang dicatat setiap bulan, prediksi semakin mendekati kenyataan.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MAIN
# ============================================================

def main():
    uploaded, tahun_sel, bulan_sel = render_sidebar()
    tgl = datetime.now().strftime('%d %B %Y')
    st.markdown(f"""
    <div class="main-header">
        <div class="header-logo">
            <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <line x1="3" y1="22" x2="21" y2="22"/>
                <line x1="6" y1="18" x2="6" y2="11"/>
                <line x1="10" y1="18" x2="10" y2="11"/>
                <line x1="14" y1="18" x2="14" y2="11"/>
                <line x1="18" y1="18" x2="18" y2="11"/>
                <polygon points="12 2 20 7 4 7"/>
            </svg>
        </div>
        <div class="header-text">
            <h1>Dashboard Keuangan Karang Taruna</h1>
            <p>Pantau keuangan desa dengan mudah, jelas, dan transparan &nbsp;·&nbsp; {tgl}</p>
        </div>
        <div class="header-badge">
            <span class="header-badge-dot"></span>Portal Manajemen Desa
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not uploaded:
        render_welcome()
        st.markdown("""
        <div class="footer-box">
            Dashboard Keuangan Karang Taruna &nbsp;·&nbsp; Transparansi &amp; Kemajuan Bersama
            <span>Upload file Excel di sidebar untuk mulai memantau keuangan Anda</span>
        </div>""", unsafe_allow_html=True)
        return

    if 'df_clean' in st.session_state:
        df = st.session_state['df_clean']
    else:
        try:
            raw = pd.read_excel(uploaded, header=0)
            if not any(k in ' '.join(raw.columns.str.lower()) for k in ['tanggal','date','tgl']):
                raw = pd.read_excel(uploaded, header=1)
            df = bersihkan_data(raw); st.session_state['df_clean'] = df
        except ValueError as ve:
            st.error(str(ve)); return
        except Exception as e:
            st.error(f"Gagal membaca file: {e}\n\nPastikan ada kolom: **Tanggal | Pemasukan | Pengeluaran**"); return

    if df.empty:
        st.warning("Data kosong atau tidak bisa dibaca. Periksa kembali format file Excel Anda."); return

    tahun_list  = sorted(df['Tahun'].unique(), reverse=True)
    tahun_aktif = tahun_sel if (tahun_sel and tahun_sel in tahun_list) else tahun_list[0]
    monthly     = agregasi_bulanan(df, tahun_aktif)
    if monthly.empty:
        st.warning(f"Tidak ada data untuk tahun {tahun_aktif}."); return

    if bulan_sel != 'Semua Bulan':
        bln_num = {v: k for k, v in MONTH_ID.items()}.get(bulan_sel)
        if bln_num:
            monthly_filtered = monthly[monthly['Bulan'] == bln_num]
            if monthly_filtered.empty:
                st.warning(f"Tidak ada data untuk bulan {bulan_sel}."); monthly_filtered = monthly
        else: monthly_filtered = monthly
    else: monthly_filtered = monthly

    pred = prediksi(monthly, tahun_aktif)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Ringkasan", "Grafik", "Prediksi", "Detail", "Tips & Saran"])

    with tab1:
        st.markdown(f'<div class="section-title">Ringkasan Keuangan — Tahun {tahun_aktif}' + (f' &nbsp;·&nbsp; {bulan_sel}' if bulan_sel != 'Semua Bulan' else '') + '</div>', unsafe_allow_html=True)
        render_kpi(monthly_filtered, pred)
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Kondisi Kesehatan Keuangan</div>', unsafe_allow_html=True)
        st.caption("Skor 0–100%  ·  Sangat Sehat (70–100%)  ·  Perlu Perhatian (40–70%)  ·  Kritis (0–40%)")
        render_health(monthly_filtered)

    with tab2:
        st.markdown(f'<div class="section-title">Grafik Keuangan — Tahun {tahun_aktif}</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(chart_saldo(monthly_filtered), use_container_width=True)
        with c2: st.plotly_chart(chart_donut(monthly_filtered['Pemasukan'].sum(), monthly_filtered['Pengeluaran'].sum()), use_container_width=True)
        c3, c4 = st.columns(2)
        with c3: st.plotly_chart(chart_bulanan(monthly_filtered), use_container_width=True)
        with c4: st.plotly_chart(chart_surplus(monthly_filtered), use_container_width=True)
        fig_kat = chart_kategori(df, tahun_aktif)
        if fig_kat: st.plotly_chart(fig_kat, use_container_width=True)
        if len(tahun_list) > 1:
            st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Perbandingan Tahun ke Tahun</div>', unsafe_allow_html=True)
            fig_yoy = chart_yoy(df, tahun_list[:4])
            if fig_yoy: st.plotly_chart(fig_yoy, use_container_width=True)

    with tab3:
        st.markdown(f'<div class="section-title">Prediksi Keuangan — {pred["bulan_nama"]} {pred["tahun"]} dan Seterusnya</div>', unsafe_allow_html=True)
        render_prediksi(pred, monthly)
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Validasi Akurasi Prediksi</div>', unsafe_allow_html=True)
        st.caption("Seberapa dekat prediksi model dengan data nyata masa lalu.")
        render_akurasi(monthly, pred)

    with tab4:
        st.markdown(f'<div class="section-title">Rekap Keuangan Bulanan — Tahun {tahun_aktif}</div>', unsafe_allow_html=True)
        st.caption("Rincian lengkap pemasukan, pengeluaran, rasio beban, dan saldo setiap bulan")
        render_tabel(monthly, tahun_aktif)
        if bulan_sel != 'Semua Bulan':
            bln_num = {v: k for k, v in MONTH_ID.items()}.get(bulan_sel)
            if bln_num:
                dd = df[(df['Tahun'] == tahun_aktif) & (df['Bulan'] == bln_num)]
                if not dd.empty:
                    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
                    st.markdown(f'<div class="section-title">Detail Transaksi: {bulan_sel} {tahun_aktif}</div>', unsafe_allow_html=True)
                    dc = ['Tanggal','Pemasukan','Pengeluaran','Saldo']
                    if 'Keterangan' in dd.columns: dc.append('Keterangan')
                    if 'Kategori'   in dd.columns: dc.append('Kategori')
                    st.dataframe(dd[dc], use_container_width=True, hide_index=True)
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Pola Pemasukan: Awal vs Akhir Bulan</div>', unsafe_allow_html=True)
        st.caption("Apakah pemasukan lebih banyak di awal bulan (tgl 1–15) atau akhir bulan (tgl 16–31)?")
        render_ab(df, tahun_aktif)

    with tab5:
        st.markdown('<div class="section-title">Saran &amp; Tips Keuangan</div>', unsafe_allow_html=True)
        render_tips(monthly, pred)

    st.markdown(f"""
    <div class="footer-box">
        Dashboard Keuangan Karang Taruna &nbsp;·&nbsp; Transparansi &amp; Kemajuan Bersama
        <span>Data Tahun {tahun_aktif} &nbsp;·&nbsp; Diperbarui {datetime.now().strftime('%d %B %Y pukul %H:%M WIB')} &nbsp;·&nbsp; v4.0</span>
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()