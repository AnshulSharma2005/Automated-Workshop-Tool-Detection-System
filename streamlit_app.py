import os
import io
import time
from pathlib import Path

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Workshop Tool Detection", page_icon="🛠️", layout="wide")

# ---------- Sidebar Controls ----------
st.sidebar.title("⚙️ Settings")
default_model_path = "runs/models/tool_detection/weights/best.pt"
model_path = st.sidebar.text_input("Model path (.pt)", value=default_model_path)
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.01)
iou = st.sidebar.slider("IoU threshold", 0.1, 0.95, 0.7, 0.01)
imgsz = st.sidebar.selectbox("Image size", [320, 416, 512, 640, 800], index=3)
device = st.sidebar.selectbox("Device", ["cpu"], index=0)  # keep CPU for your machine

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Keep images under ~5–8MB for snappy results.")

# ---------- Cache the model so it loads once ----------
@st.cache_resource(show_spinner=True)
def load_model(weights_path: str):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model not found: {weights_path}")
    return YOLO(weights_path)

# ---------- Helpers ----------
def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def pil_bytes(img_rgb: np.ndarray, fmt="JPEG") -> bytes:
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt, quality=95)
    return buf.getvalue()

def run_image_inference(model: YOLO, image: Image.Image, conf: float, iou: float, imgsz: int):
    # Convert PIL -> numpy BGR for Ultralytics plotting
    img_rgb = np.array(image.convert("RGB"))
    results = model.predict(
        img_rgb,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False
    )
    annotated_bgr = results[0].plot()  # returns BGR
    annotated_rgb = bgr_to_rgb(annotated_bgr)
    return results[0], annotated_rgb

def save_uploaded_file(upload, suffix: str) -> Path:
    out_dir = Path("streamlit_tmp")
    out_dir.mkdir(exist_ok=True, parents=True)
    fp = out_dir / f"{int(time.time()*1000)}_{upload.name}"
    with open(fp, "wb") as f:
        f.write(upload.getbuffer())
    return fp

def run_video_inference(model: YOLO, video_path: Path, conf: float, iou: float, imgsz: int) -> Path:
    # Save outputs under a dedicated project/name so we can find the rendered video
    project = "streamlit_outputs"
    name = f"pred_{int(time.time())}"
    results = model.predict(
        source=str(video_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        save=True,
        project=project,
        name=name,
        verbose=False
    )
    # Ultralytics saves processed media in project/name/
    out_dir = Path(project) / name
    # Find first video file in output dir
    for ext in (".mp4", ".avi", ".mov", ".mkv"):
        cand = list(out_dir.glob(f"*{ext}"))
        if cand:
            return cand[0]
    # fallback: return any file saved
    files = list(out_dir.iterdir())
    return files[0] if files else None

# ---------- UI ----------
st.title("🛠️ Automated Workshop Tool Detection — Web App")

col_l, col_r = st.columns([1, 1])
with col_l:
    st.subheader("Upload Image")
    up_img = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="img_up")

with col_r:
    st.subheader("Upload Video (optional)")
    up_vid = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"], key="vid_up")

# Load model (once)
with st.spinner("Loading model..."):
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Unable to load model. Check path.\n\n{e}")
        st.stop()

# ---------- Image flow ----------
if up_img is not None:
    image = Image.open(up_img)
    st.image(image, caption="Uploaded", use_container_width=True)

    if st.button("Run detection on image", type="primary"):
        with st.spinner("Running detection..."):
            res, annotated_rgb = run_image_inference(model, image, conf, iou, imgsz)

        # Show detections
        st.subheader("Detections")
        st.image(annotated_rgb, caption="Result", use_container_width=True)

        # Class-wise summary
        if res.boxes is not None and len(res.boxes) > 0:
            names = model.model.names if hasattr(model.model, "names") else {}
            counts = {}
            for cls_idx in res.boxes.cls.cpu().numpy().astype(int):
                label = names.get(cls_idx, str(cls_idx))
                counts[label] = counts.get(label, 0) + 1
            st.write("**Objects detected:**")
            st.json(counts)
        else:
            st.info("No objects detected.")

        # Download button
        out_bytes = pil_bytes(annotated_rgb, fmt="JPEG")
        st.download_button(
            "Download result image",
            data=out_bytes,
            file_name="detection_result.jpg",
            mime="image/jpeg"
        )

# ---------- Video flow ----------
if up_vid is not None:
    if st.button("Run detection on video"):
        with st.spinner("Processing video — this may take a moment..."):
            tmp_path = save_uploaded_file(up_vid, suffix="video")
            out_video = run_video_inference(model, tmp_path, conf, iou, imgsz)

        if out_video and out_video.exists():
            st.success("Done!")
            st.video(str(out_video))
            with open(out_video, "rb") as f:
                st.download_button(
                    "Download result video",
                    data=f.read(),
                    file_name=out_video.name,
                    mime="video/mp4"
                )
        else:
            st.error("Could not locate output video. Check logs or try a shorter clip.")

st.markdown("---")
st.caption("Built with Streamlit + Ultralytics YOLOv8")
