import streamlit as st
from PIL import Image
from pathlib import Path

# ========================================
# KONFIGURASI HALAMAN
# ========================================
st.set_page_config(
    page_title="🏠 Beranda",
    layout="wide",
)

# ========================================
# CSS GLOBAL + PRINT PDF
# ========================================
st.markdown("""
<style>
/* Hindari pecah halaman saat cetak PDF */
.no-page-break {
    page-break-inside: avoid;
    break-inside: avoid;
}

@media print {
    .no-page-break {
        page-break-inside: avoid !important;
        break-inside: avoid !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ========================================
# SIDEBAR
# ========================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 5px;">
        <h2 style="margin-bottom: 0; color:#d9534f;">🖐️ Hand Detection App</h2>
        <p style="margin-top: -5px; color:#666;">Deteksi Angka pada Tangan</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.page_link("Home.py", label="🏠 Beranda")
    st.page_link("pages/1_Deteksi_Realtime.py", label="📸 Deteksi Realtime")
    st.page_link("pages/2_Deteksi_GradCAM.py", label="🔥 Deteksi Grad-CAM")

    for _ in range(20):
        st.write("")

    st.markdown("---")
    st.markdown("""
        <div style="font-size: 12px; color:#555;">
            Skripsi 2025 - H071221025 <br>
            Universitas Hasanuddin
        </div>
    """, unsafe_allow_html=True)

# ========================================
# HEADER
# ========================================
st.markdown("""
<div style='padding: 22px; border-radius: 15px; border: 1px solid #e6e6e6;'>
    <h1 style='margin-bottom: 5px;'>📊 Dashboard Deteksi Angka pada Tangan</h1>
    <p>
        Sistem deteksi angka menggunakan computer vision dan deep learning 
        untuk mengenali angka pada gestur tangan.
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# 🔴 BUKA CONTAINER (GAMBAR + MULAI PREDIKSI)
# =====================================================
st.markdown("""
<div class="no-page-break">
""", unsafe_allow_html=True)

# ========================================
# REFERENSI GAMBAR
# ========================================
st.subheader("Referensi Bentuk Tangan (0–9)")
st.caption("Contoh bentuk tangan untuk setiap angka.")


BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"

image_paths = [
    ASSETS_DIR / "angka_0.jpg",
    ASSETS_DIR / "angka_1.jpg",
    ASSETS_DIR / "angka_2.jpg",
    ASSETS_DIR / "angka_3.jpg",
    ASSETS_DIR / "angka_4.jpg",
    ASSETS_DIR / "angka_5.jpg",
    ASSETS_DIR / "angka_6.jpg",
    ASSETS_DIR / "angka_7.jpg",
    ASSETS_DIR / "angka_8.jpg",
    ASSETS_DIR / "angka_9.jpg",
]


labels = [f"Angka {i}" for i in range(10)]

def image_card(path, label):
    if path.exists():
        img = Image.open(path)
        st.image(img, use_column_width=True)
    else:
        st.error(f"Gambar tidak ditemukan: {path.name}")

    st.markdown(
        f"<p style='text-align:center; font-weight:600;'>{label}</p>",
        unsafe_allow_html=True
    )


# Baris 1
cols = st.columns(5)
for i in range(5):
    with cols[i]:
        image_card(image_paths[i], labels[i])

# Spasi kecil (AMAN PDF)
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# Baris 2
cols2 = st.columns(5)
for i in range(5, 10):
    with cols2[i - 5]:
        image_card(image_paths[i], labels[i])

# ========================================
# MULAI PREDIKSI (NEMPEL DI BAWAH GAMBAR)
# ========================================
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
st.subheader("Mulai Prediksi")

st.markdown("""
<style>
.detect-card, .gradcam-card {
    padding: 26px;
    border-radius: 18px;
    text-align: center;
    color: white;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.detect-card {
    background: linear-gradient(135deg, #0099cc, #00ccaa);
}
.gradcam-card {
    background: linear-gradient(135deg, #ff6f91, #ff9671);
}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

col1.markdown("""
<a href="/Deteksi_Realtime" target="_self" style="text-decoration:none;">
<div class="detect-card">
    <h2>📸 Deteksi Realtime</h2>
    <p>Mulai deteksi gestur tangan dengan webcam</p>
</div>
</a>
""", unsafe_allow_html=True)

col2.markdown("""
<a href="/Deteksi_GradCAM" target="_self" style="text-decoration:none;">
<div class="gradcam-card">
    <h2>🔥 Deteksi Grad-CAM</h2>
    <p>Visualisasi fokus model pada prediksi</p>
</div>
</a>
""", unsafe_allow_html=True)

# =====================================================
# 🔴 TUTUP CONTAINER (INI YANG KAMU TANYAKAN)
# =====================================================
st.markdown("""
</div>
""", unsafe_allow_html=True)
