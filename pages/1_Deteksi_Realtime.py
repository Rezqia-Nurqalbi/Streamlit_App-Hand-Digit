import streamlit as st
import cv2
import numpy as np
import time
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16, ResNet50
import mediapipe as mp
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==========================
# KONFIGURASI
# ==========================
st.set_page_config(page_title="📸 Deteksi Realtime", layout="wide")
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 10
MODEL_FOLDER_PATH = r"C:\Users\hp_fq\OneDrive\Documents\Skripsi Penelitian\Model Baru"
PERMANENT_HAND_CONF = 0.5


# ==========================
# SIDEBAR
# ==========================
with st.sidebar:
    st.markdown("""
        <div style="padding: 10px; margin-bottom: -10px">
            <h2 style="margin:0; color:#d9534f;">🖐️ Hand Detection App</h2>
            <p style="margin-top:-5px; color:#aaa;">Deteksi Angka pada Tangan</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.page_link("Home.py", label="🏠 Beranda")
    st.page_link("pages/1_Deteksi_Realtime.py", label="📸 Deteksi Realtime")
    st.page_link("pages/2_Deteksi_GradCAM.py", label="🔥 Deteksi Grad-CAM")

    # ====== SPACER OTOMATIS ======
    # Spacer sebanyak mungkin agar footer turun ke paling bawah
    for _ in range(20):
        st.write("")

    st.markdown("---")
    # ====== FOOTER PALING BAWAH ======
    st.markdown("""
        <div style="font-size: 12px; color:#555;">
            Skripsi 2025 - H071221025 <br>
            Universitas Hasanuddin
        </div>
    """, unsafe_allow_html=True)


# ==========================
# Ambil daftar model di folder
# ==========================
def get_model_paths(folder):
    models = {}
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if f.endswith(".keras"):
                models[f] = os.path.join(folder, f)
    return models


# ==========================
# Build Base Model
# ==========================
def build_base_model(name_hint):
    name_hint = name_hint.lower()
    base = ResNet50(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE) \
        if "resnet" in name_hint \
        else VGG16(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)

    base.trainable = False
    return keras.Sequential([
        base,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])


# ==========================
# Load model single
# ==========================
@st.cache_resource
def load_model_safe(model_name, path):
    try:
        return keras.models.load_model(path, compile=False), f"Berhasil Memuat Model {model_name}"
    except:
        try:
            model = build_base_model(model_name)
            model.load_weights(path, by_name=True, skip_mismatch=True)
            return model, f"{model_name}"
        except Exception as e:
            return None, f"Error memuat {model_name}: {e}"


# ==========================
# PREPROCESS
# ==========================
def preprocess(img):
    img = img.convert('RGB').resize((224, 224))
    arr = np.array(img).astype('float32')
    return np.expand_dims(arr, 0)

def predict(model, img):
    p = model.predict(preprocess(img), verbose=0)
    return int(np.argmax(p)), p[0]


# ==========================
# HEADER
# ==========================
st.markdown("""
<h1 style="color:#c0392b;">Deteksi RealTime</h1>
<p style="color:#777;">Gunakan Kamera untuk mendeteksi angka yang Anda tunjukkan dengan jari tangan secara real-time</p>
""", unsafe_allow_html=True)


# ==========================
# 🔧 Pengaturan Model dengan placeholder pesan di atas dropdown
# ==========================
st.markdown("### 🔧 Pengaturan Model")

with st.expander("Pilih Model (.keras):", expanded=True):

    # Ambil daftar model
    model_paths = get_model_paths(MODEL_FOLDER_PATH)

    if not model_paths:
        st.error(f"Tidak ada file model '.keras' di folder:\n{MODEL_FOLDER_PATH}")
        st.stop()

    # Placeholder pesan muncul di atas dropdown
    model_message = st.empty()

    # Dropdown model
    model_selected = st.selectbox(
        "Pilih Model:",
        ["-- pilih model --"] + list(model_paths.keys())
    )

    # Tombol load model
    load_btn = st.button("📥 Muat Model")
    if load_btn:
        if model_selected == "-- pilih model --":
            model_message.warning("Pilih model terlebih dahulu!")
        else:
            with st.spinner("Memuat model..."):
                model, msg = load_model_safe(
                    model_selected,
                    model_paths[model_selected]
                )
                if model:
                    st.session_state.model = model
                    model_message.success(f"{msg}")
                else:
                    model_message.error(f"{msg}")

# ==========================
# Session State
# ==========================
if "model" not in st.session_state:
    st.session_state.model = None
if "running" not in st.session_state:
    st.session_state.running = False
if "probs" not in st.session_state:
    st.session_state.probs = np.zeros(NUM_CLASSES)

# ==========================
# Layout Kolom (Video + Hasil)
# ==========================
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("🖐️ Kamera")
    st.markdown("Silakan posisikan tangan Anda di depan kamera secara jelas untuk memulai deteksi.")
    cam_placeholder = st.empty()

    # ---------------------
    # Placeholder kamera OFF
    # ---------------------
    if not st.session_state.running:
        cam_placeholder.markdown(""" 
            <div style="width:100%; height:430px; background:#0f172a; border-radius:14px;
                        display:flex; align-items:center; justify-content:center; color:#94a3b8;
                        box-shadow:0 4px 12px rgba(0,0,0,0.25);">
                <div style="text-align:center;">
                    <div style="font-size:40px;">📷</div>
                    <div style="font-size:22px;font-weight:600;">Kamera Nonaktif</div>
                    <div style="font-size:14px;opacity:0.7;">Tekan tombol Mulai untuk menyalakan kamera</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ---------------------
    # Spacer sebelum tombol
    # ---------------------
    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    # Tombol mulai / Stop
    c1, c2 = st.columns(2)
    mulai_btn = c1.button("▶️ Mulai", use_container_width=True)
    stop_btn = c2.button("⏹️ Henti", use_container_width=True)

    fps_text = st.empty()

# ---------------------
# Barchart default (tetap ada saat kamera OFF)
# ---------------------
with col2:
    st.subheader("🔢 Hasil Deteksi")
    st.markdown("Barchart menampilkan probabilitas prediksi angka. Biru untuk tangan pertama, oranye untuk tangan kedua.")
    result_text = st.empty()
    chart_area = st.empty()

    if not st.session_state.running:
        fig, ax = plt.subplots(figsize=(5, 4))
        x = np.arange(NUM_CLASSES)
        ax.bar(x, np.zeros(NUM_CLASSES))
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(NUM_CLASSES)])
        ax.set_ylim(0, 1)
        chart_area.pyplot(fig)


# ==========================
# Placeholder Kamera OFF
# ==========================
if not st.session_state.running:
    with col1:
        cam_placeholder.markdown(""" 
            <div style="width:100%; height:430px; background:#0f172a; border-radius:14px;
                        display:flex; align-items:center; justify-content:center; color:#94a3b8;
                        box-shadow:0 4px 12px rgba(0,0,0,0.25);">
                <div style="text-align:center;">
                    <div style="font-size:40px;">📷</div>
                    <div style="font-size:22px;font-weight:600;">Kamera Nonaktif</div>
                    <div style="font-size:14px;opacity:0.7;">Tekan tombol Mulai untuk menyalakan kamera</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(NUM_CLASSES)
    ax.bar(x, np.zeros(NUM_CLASSES))
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(NUM_CLASSES)])
    ax.set_ylim(0, 1)
    chart_area.pyplot(fig)

def render_camera_off():
    # Kamera inactive box
    cam_placeholder.markdown(""" 
        <div style="width:100%; height:430px; background:#0f172a; border-radius:14px;
                    display:flex; align-items:center; justify-content:center; color:#94a3b8;
                    box-shadow:0 4px 12px rgba(0,0,0,0.25);">
            <div style="text-align:center;">
                <div style="font-size:40px;">📷</div>
                <div style="font-size:22px;font-weight:600;">Kamera Nonaktif</div>
                <div style="font-size:14px;opacity:0.7;">Tekan tombol Mulai untuk menyalakan kamera</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Barchart default
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(NUM_CLASSES)
    ax.bar(x, np.zeros(NUM_CLASSES))
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(NUM_CLASSES)])
    ax.set_ylim(0, 1)
    chart_area.pyplot(fig)

# -----------------------
# Tombol mulai / Stop
# -----------------------
if mulai_btn:
    if st.session_state.model is None:
        model_message.error("❌ Model belum dimuat!")
        st.session_state.running = False
        render_camera_off()  # tampilkan kembali box + chart
    else:
        st.session_state.running = True

if stop_btn:
    st.session_state.running = False
    render_camera_off()  # tampilkan kembali box + chart



# ==========================
# CAMERA LOOP
# ==========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ==========================
# Session State ekstra
# ==========================
if "probs_tangan1" not in st.session_state:
    st.session_state.probs_tangan1 = np.zeros(NUM_CLASSES)
if "probs_tangan2" not in st.session_state:
    st.session_state.probs_tangan2 = np.zeros(NUM_CLASSES)


# ==========================
# CAMERA LOOP DENGAN 2 TANGAN
# ==========================
def run_camera():

    if st.session_state.model is None:
        st.error("Pilih model terlebih dahulu.")
        st.session_state.running = False
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Kamera tidak ditemukan.")
        st.session_state.running = False
        return

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=PERMANENT_HAND_CONF,
        min_tracking_confidence=0.5
    )

    prev = time.time()

    while st.session_state.running:
        detected_Tangan1 = False
        detected_Tangan2 = False
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        res = hands.process(rgb)

        # Reset sementara

        if res.multi_hand_landmarks:

            for idx, hand in enumerate(res.multi_hand_landmarks):

                xs = [lm.x for lm in hand.landmark]
                ys = [lm.y for lm in hand.landmark]
                x1, x2 = int(min(xs)*w)-20, int(max(xs)*w)+20
                y1, y2 = int(min(ys)*h)-20, int(max(ys)*h)+20

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                roi = rgb[y1:y2, x1:x2]

                if roi.size > 0:
                    img = Image.fromarray(roi)
                    cls, probs = predict(st.session_state.model, img)

                    # kiri = idx 0 → biru
                    # kanan = idx 1 → oranye
                    if idx == 0:
                        st.session_state.probs_tangan1 = probs
                        color = (255, 100, 0)  # biru-ish BGR
                        text = f"Tangan1: {cls} ({probs[cls]*100:.1f}%)"
                        detected_Tangan1 = True
                    else:
                        st.session_state.probs_tangan2 = probs
                        color = (0, 140, 255)  # oranye
                        text = f"Tangan2: {cls} ({probs[cls]*100:.1f}%)"
                        detected_Tangan2 = True

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # Tampilkan frame
        cam_placeholder.image(frame, channels="BGR")

        if not detected_Tangan1:
            st.session_state.probs_tangan1 = np.zeros(NUM_CLASSES)
        if not detected_Tangan2:
            st.session_state.probs_tangan2 = np.zeros(NUM_CLASSES)
        # ======================
        # BAR CHART DUA TANGAN
        # ======================
        fig, ax = plt.subplots(figsize=(5, 4))

        x = np.arange(NUM_CLASSES)

        ax.bar(x - 0.15, st.session_state.probs_tangan1,
               width=0.3, label="Tangan1", color="blue")
        ax.bar(x + 0.15, st.session_state.probs_tangan2,
               width=0.3, label="Tangan2", color="orange")

        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(NUM_CLASSES)])

        ax.set_ylim(0, 1)
        ax.legend()

        chart_area.pyplot(fig)

        # FPS
        now = time.time()
        fps_text.text(f"FPS: {1/(now-prev):.1f}")
        prev = now

        time.sleep(0.01)

    cap.release()
    hands.close()

# ==========================
# Eksekusi Kamera
# ==========================
if st.session_state.running:
    run_camera()