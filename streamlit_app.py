import os
import io
import time
from pathlib import Path

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

# Page Setup
st.set_page_config(
    page_title="Workshop Tool Detection",
    page_icon="🛠️",
    layout="wide",
)

st.markdown("""
<style>
/* Reduce container width and padding */
.main > div {
    padding-top : 1rem;
}

/* Compact image preview box */
.preview-box {
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 6px;
    background: #fafafa;
}

/* Buttons centered */
.stButton>button {
    width : 100%;
    border-radius: 8px;
    font-weight: 600;
}

/* Section header style */
h2, h3 {
    margin-bottom : 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Model Settings")
default_model_path = "runs/detect/train12/weights/best.pt"

model_path = st.sidebar.text_input("Model path", value=default_model_path)
conf = st.sidebar.slider("Confidence", 0.05, 0.95, 0.25, 0.01)
iou = st.sidebar.slider("IoU Threshold", 0.1, 0.95, 0.7, 0.01)
imgsz = st.sidebar.selectbox("Image Size", [320, 416, 512, 640, 800], index=3)
device = st.sidebar.selectbox("Device", ["cpu"], index=0)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Use clear, bright images for best results.")

# Model Cache
@st.cache_resource(show_spinner=True)
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found.")
    return YOLO(model_path)

# Helper functions
def bgr_to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def pil_bytes(img, fmt="JPEG"):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format=fmt, quality=95)
    return buf.getvalue()

def run_image_inference(model, image, conf, iou, imgsz):
    img = np.array(image)
    res = model.predict(img, conf=conf, iou=iou, imgsz=imgsz, device=device)[0]
    annotated = bgr_to_rgb(res.plot())
    return res, annotated

# Title
st.title("🛠️ Workshop Tool Detection Dashboard")

# Two input panels (side-by-side)
col_upload_img, col_upload_vid = st.columns([1, 1])

with col_upload_img:
    st.subheader("📷 Upload Image")
    up_img = st.file_uploader("Select image", type=["jpg", "jpeg", "png"])

with col_upload_vid:
    st.subheader("🎞️ Upload Video")
    up_vid = st.file_uploader("Select video", type=["mp4", "avi", "mov", "mkv"])

# Load model once
with st.spinner("Loading YOLO model..."):
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(str(e))
        st.stop()

# IMAGE INFERENCE SECTION
if up_img:
    img = Image.open(up_img)

    col_show, col_detect = st.columns([1, 1])

    with col_show:
        st.subheader("📌 Uploaded Image")
        st.image(img, use_container_width=True, caption="Original")

    with col_detect:
        st.subheader("🔍 Detection")
        if st.button("Run Image Detection", type="primary"):
            with st.spinner("Analyzing image..."):
                res, annotated = run_image_inference(model, img, conf, iou, imgsz)
            st.image(annotated, use_container_width=True, caption="Detected Objects")

            # Class count summary
            if res.boxes is not None and len(res.boxes) > 0:
                names = model.model.names
                counts = {}
                for cls_id in res.boxes.cls.int().tolist():
                    label = names.get(cls_id, str(cls_id))
                    counts[label] = counts.get(label, 0) + 1
                st.success("📊 Objects Found")
                st.json(counts)
            else:
                st.info("No objects detected.")

            # Download button
            st.download_button(
                "Download Annotated Image",
                data=pil_bytes(annotated),
                file_name="result.jpg",
                mime="image/jpeg",
            )

# VIDEO INFERENCE SECTION
if up_vid:
    if st.button("Run Video Detection"):
        st.warning("⚠️ Processing video... may take time on CPU.")
        tmp_path = Path("uploaded_video.mp4")
        with open(tmp_path, "wb") as f:
            f.write(up_vid.getbuffer())

        with st.spinner("Running YOLO on video..."):
            project = "streamlit_outputs"
            name = f"pred_{int(time.time())}"

            model.predict(
                source=str(tmp_path),
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                save=True,
                project=project,
                name=name,
                verbose=False,
            )

        out_path = Path(project) / name
        files = list(out_path.glob("*.mp4"))

        if files:
            st.video(str(files[0]))
            with open(files[0], "rb") as f:
                st.download_button(
                    "Download Processed Video",
                    data=f.read(),
                    file_name="predicted_video.mp4",
                    mime="video/mp4"
                )
        else:
            st.error("Could not generate output video.")

st.markdown("---")
st.caption("Built with using Streamlit & YOLOv8")
