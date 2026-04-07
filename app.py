import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# =====================
# CONFIG
# =====================
st.set_page_config(page_title="Pothole Detection", layout="wide")

# Hide Streamlit default menu & footer
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =====================
# LOAD MODEL
# =====================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =====================
# TITLE
# =====================
st.markdown("""
<h1 style='text-align: center;'>Pothole Detection System</h1>
<p style='text-align: center; color: gray;'>Upload image atau gunakan kamera untuk mendeteksi jalan berlubang</p>
""", unsafe_allow_html=True)

# =====================
# DETECTION FUNCTION
# =====================
def detect_image(image):
    results = model(image)
    annotated = results[0].plot()
    detected = len(results[0].boxes) > 0
    return annotated, detected

# =====================
# REALTIME CAMERA CLASS
# =====================
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        annotated = results[0].plot()

        # Tambahin status di frame
        if len(results[0].boxes) > 0:
            label = "PERLU DIPERBAIKI"
        else:
            label = "AMAN"

        # Tampilkan text di frame
        import cv2
        cv2.putText(annotated, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return annotated

# =====================
# LAYOUT
# =====================
tab1, tab2 = st.tabs(["Upload Image", "Camera"])

# =====================
# TAB UPLOAD
# =====================
with tab1:
    uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image", width='stretch')

        with col2:
            with st.spinner("Detecting..."):
                result, detected = detect_image(image_np)
                st.image(result, caption="Detection Result", width='stretch')

                if detected:
                    st.error("Jalan perlu diperbaiki")
                else:
                    st.success("Jalan tidak perlu diperbaiki")

# =====================
# TAB CAMERA (REALTIME)
# =====================
with tab2:
    webrtc_streamer(
        key="pothole-detection",
        video_transformer_factory=VideoTransformer,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )

# =====================
# FOOTER
# =====================
st.markdown("""
<hr>
<p style='text-align: center; color: gray;'>Pothole Detection App</p>
""", unsafe_allow_html=True)
