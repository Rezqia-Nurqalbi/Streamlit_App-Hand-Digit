import streamlit as st
import cv2
import numpy as np
import time
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================
# CONFIG
# ============================
st.set_page_config(page_title="🔥 Deteksi Grad-CAM", layout="wide")

INPUT_SHAPE = (224,224,3)
NUM_CLASSES = 10
MODEL_FOLDER_PATH = r"C:\Users\hp_fq\OneDrive\Documents\Skripsi Penelitian\Model Baru"

# =============== GANTI SESUAI FILE KAMU =================
PLACEHOLDER_IMAGE_PATH = "/mnt/data/Screenshot_22-11-2025_185211_localhost.jpeg"
# ==============================================================


# ============================
# SIDEBAR
# ============================
with st.sidebar:
    st.markdown("""
        <div style="padding: 10px; margin-bottom: -10px">
            <h2 style="margin: 0; color:#d9534f;">🖐️ Hand Detection App</h2>
            <p style="margin-top: -5px; color:#aaa;">Deteksi Angka pada Tangan</p>
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


# ============================
# FUNGSIONALITAS MODEL
# ============================
def get_model_paths_from_folder(folder_path):
    models = {}
    if not os.path.exists(folder_path):
        return models
    for filename in os.listdir(folder_path):
        if filename.endswith(".keras"):
            models[filename] = os.path.join(folder_path, filename)
    return models

@st.cache_resource
def load_model_safe(model_name: str, path: str):
    try:
        model = keras.models.load_model(path, compile=False)
        return model, f"Berhasil Memuat Model '{model_name}'"
    except Exception as e:
        return None, f"Error: {e}"

def preprocess_image(img: Image.Image):
    img = img.convert('RGB').resize((224,224))
    arr = np.array(img).astype('float32')
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(model, img: Image.Image):
    preds = model.predict(preprocess_image(img), verbose=0)
    cls = int(np.argmax(preds))
    prob = float(np.max(preds))
    return cls, prob

def get_base_and_last_conv(model, name_hint):
    name_hint = name_hint.lower()
    if "resnet" in name_hint:
        base_model = model.get_layer("resnet50")
        last_conv_layer = base_model.get_layer("conv5_block3_out")
    else:
        base_model = model.get_layer("vgg16")
        last_conv_layer = base_model.get_layer("block5_conv3")
    return base_model, last_conv_layer

def make_gradcam_heatmap(img_array, base_model, last_conv_layer):
    grad_model = tf.keras.models.Model([base_model.input], last_conv_layer.output)
    with tf.GradientTape() as tape:
        conv_outputs = grad_model(img_array)

    pooled = tf.reduce_mean(conv_outputs, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam_on_image(img_pil, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img_pil.width, img_pil.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    final_img = cv2.addWeighted(np.array(img_pil), 1 - alpha, heatmap, alpha, 0)
    return final_img

# =======================
# HEADER UTAMA
# =======================
st.markdown("""
    <h1 style="color:#c0392b; margin-bottom: -5px;">
        Deteksi Grad-Cam
    </h1>
    <p style="color:#777; font-size:16px;">
        Lihat area mana yang difokuskan oleh model saat mendeteksi angka menggunakan Gradient-weighted Class Activation Mapping
    </p>
""", unsafe_allow_html=True)

# ============================
# PENGATURAN MODEL (UI) — FIX TANPA ERROR
# ============================
st.markdown("### 🔧 Pengaturan Model")

with st.expander("Pilih Model (.keras):", expanded=True):

    model_paths = get_model_paths_from_folder(MODEL_FOLDER_PATH)
    if not model_paths:
        st.error(f"❌ Tidak ada model '.keras' di folder:\n{MODEL_FOLDER_PATH}")
        st.stop()

    # ---- Placeholder aman (TIDAK disimpan di session_state) ----
    model_message = st.empty()

    model_selected = st.selectbox("Pilih Model:", ["-- pilih model --"] + list(model_paths.keys()))
    load_btn = st.button("📥 Muat Model")

    if load_btn:
        if model_selected == "-- pilih model --":
            model_message.warning("Pilih model terlebih dahulu!")
        else:
            with st.spinner("Memuat model..."):
                model, msg = load_model_safe(model_selected, model_paths[model_selected])
                if model:
                    st.session_state["models"] = {model_selected: model}
                    model_message.success(f"{msg}")
                else:
                    model_message.error(f"❌ {msg}")


# ============================
# KAMERA + GRADCAM AREA
# ============================

col1, col2 = st.columns([3,2])

with col1:
    st.subheader("🖐️ Kamera")
    st.markdown("Silakan posisikan tangan Anda di depan kamera secara jelas untuk memulai deteksi.")
    cam_box = st.empty()
    fps_box = st.empty()

with col2:
    st.subheader("🔥 Grad-CAM Heatmap")
    st.markdown("Visualisasi area fokus model")
    gradcam_box = st.empty()


# ============================
# PLACEHOLDER SEBELUM MULAI
# ============================
if "running" not in st.session_state:
    st.session_state["running"] = False

if not st.session_state["running"]:
    with col1:
        if os.path.exists(PLACEHOLDER_IMAGE_PATH):
            img = Image.open(PLACEHOLDER_IMAGE_PATH)
            cam_box.image(img, caption="Kamera Nonaktif — Tekan Mulai", use_container_width=True)
        else:
            cam_box.markdown("""
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

    with col2:
        gradcam_box.markdown("""
            <div style="
                width:100%;height:430px;background:#111827;border-radius:12px;
                display:flex;align-items:center;justify-content:center;color:#6b7280;">
                <div style="text-align:center;">
                    <div style="font-size:26px;">🔥 Grad-CAM Nonaktif</div>
                    <div style="font-size:14px;">Aktif setelah kamera berjalan</div>
                </div>
            </div>
        """, unsafe_allow_html=True)


# ============================
# TOMBOL START/STOP
# ============================
st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
colA, colB = st.columns(2)


with colA:
    if st.button("▶️ Mulai", use_container_width=True):
        if not st.session_state.get("models"):
            model_message.error("❌ Model belum dimuat!")
            st.session_state["running"] = False
        else:
            st.session_state["running"] = True

with colB:
    if st.button("⏹️ Henti", use_container_width=True):
        st.session_state["running"] = False


# ============================
# JIKA TIDAK RUNNING → TAMPILKAN PLACEHOLDER
# ============================
if not st.session_state["running"]:

    # --- kamera inactive ---
    if os.path.exists(PLACEHOLDER_IMAGE_PATH):
        img = Image.open(PLACEHOLDER_IMAGE_PATH)
        cam_box.image(img, caption="Kamera Nonaktif — Tekan Mulai", use_container_width=True)
    else:
        cam_box.markdown("""
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

    # --- gradcam inactive ---
    gradcam_box.markdown("""
        <div style="
            width:100%;height:430px;background:#111827;border-radius:12px;
            display:flex;align-items:center;justify-content:center;color:#6b7280;">
            <div style="text-align:center;">
                <div style="font-size:26px;">🔥 Grad-CAM Nonaktif</div>
                <div style="font-size:14px;">Aktif setelah kamera berjalan</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # hentikan loop kamera
    st.stop()



# ============================
# LOOP KAMERA — TANPA ERROR
# ============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def camera_loop():
    models_loaded = st.session_state["models"]

    model_name = list(models_loaded.keys())[0]
    model = models_loaded[model_name]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Tidak bisa membuka kamera!")
        return

    prev_time = None

    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        try:
            while st.session_state["running"] and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                gradcam_imgs = []  # menampung Grad-CAM tiap tangan

                if results.multi_hand_landmarks:
                    for idx, hand in enumerate(results.multi_hand_landmarks):
                        h, w, _ = frame.shape
                        xs = [lm.x for lm in hand.landmark]
                        ys = [lm.y for lm in hand.landmark]

                        x1, x2 = int(min(xs)*w)-10, int(max(xs)*w)+10
                        y1, y2 = int(min(ys)*h)-10, int(max(ys)*h)+10

                        x1, y1 = max(x1, 0), max(y1, 0)
                        x2, y2 = min(x2, w), min(y2, h)

                        roi = frame[y1:y2, x1:x2]

                        if roi.size != 0:
                            img_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                            cls, prob = predict_image(model, img_pil)
                            base, last = get_base_and_last_conv(model, model_name)
                            arr = preprocess_image(img_pil)
                            heatmap = make_gradcam_heatmap(arr, base, last)
                            gradcam_img = overlay_gradcam_on_image(img_pil, heatmap)

                            # simpan Grad-CAM tiap tangan
                            gradcam_imgs.append(gradcam_img)

                            # overlay bounding box di frame kamera
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{cls} ({prob*100:.1f}%)", (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                # Tampilkan frame kamera
                cam_box.image(frame, channels="BGR")

                # Tampilkan Grad-CAM gabungan (dua tangan berdampingan)
                if gradcam_imgs:
                    # pastikan semua Grad-CAM dalam bentuk PIL.Image
                    pil_imgs = [img if isinstance(img, Image.Image) else Image.fromarray(img) for img in gradcam_imgs]

                    total_width = sum(img.width for img in pil_imgs)
                    max_height = max(img.height for img in pil_imgs)
                    combined = Image.new("RGB", (total_width, max_height))
                    x_offset = 0
                    for img in pil_imgs:
                        combined.paste(img, (x_offset, 0))
                        x_offset += img.width

                    gradcam_box.image(combined)
                    
                else:
                    gradcam_box.markdown("""<div style='width:100%;height:430px;background:#111827;
                        border-radius:12px;display:flex;align-items:center;justify-content:center;
                        color:#6b7280;'>
                        <div style='text-align:center;'>
                            <div style='font-size:26px;'>🔥 Grad-CAM Nonaktif</div>
                            <div style='font-size:14px;'>Tidak ada tangan terdeteksi</div>
                        </div>
                    </div>""", unsafe_allow_html=True)


                # FPS
                curr = time.time()
                fps = 1 / (curr - prev_time) if prev_time else 0
                prev_time = curr
                fps_box.text(f"FPS: {fps:.1f}")

                time.sleep(0.01)

        finally:
            cap.release()
            st.session_state["running"] = False

if st.session_state["running"]:
    camera_loop()

