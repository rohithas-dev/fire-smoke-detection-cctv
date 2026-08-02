import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import tempfile

# ---------- CONFIG ----------
MODEL_PATH = "fire-and-smoke-detection-yolov8/weights/best.pt"  # <-- put YOUR model's actual path here
VALID_USERS = {
    "admin": "fire123",
    "teacher": "demo2024",
    "guest": "guest123"
}
FRAME_SKIP = 5                    # process every 5th frame (speeds things up)
CONF_THRESHOLD = 0.4
# -----------------------------

st.set_page_config(page_title="Fire & Smoke CCTV Detection", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.title("🔥 CCTV Fire & Smoke Detection System")
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in VALID_USERS and VALID_USERS[username] == password:
            st.session_state.logged_in = True
            st.session_state.current_user = username
            st.rerun()
        else:
            st.error("Invalid username or password")

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_idx = 0
    records = []
    first_fire_frame = None
    first_fire_time = None
    keyframes = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP == 0:
            results = model(frame, conf=CONF_THRESHOLD, verbose=False)
            boxes = results[0].boxes
            timestamp = frame_idx / fps

            fire_count = sum(1 for c in boxes.cls if model.names[int(c)] == "fire")
            smoke_count = sum(1 for c in boxes.cls if model.names[int(c)] == "smoke")
            max_conf = float(boxes.conf.max()) if len(boxes) > 0 else 0.0

            records.append({
                "frame": frame_idx,
                "time_sec": round(timestamp, 2),
                "fire_detections": fire_count,
                "smoke_detections": smoke_count,
                "max_confidence": max_conf
            })

            if (fire_count > 0 or smoke_count > 0) and first_fire_frame is None:
                first_fire_frame = frame_idx
                first_fire_time = timestamp
                keyframes["onset"] = results[0].plot()

        frame_idx += 1

    cap.release()
    df = pd.DataFrame(records)
    return {"df": df, "first_fire_frame": first_fire_frame,
            "first_fire_time": first_fire_time, "keyframes": keyframes}

def display_results(data):
    df = data["df"]
    first_fire_frame = data["first_fire_frame"]
    first_fire_time = data["first_fire_time"]
    keyframes = data["keyframes"]

    st.header("📋 Detection Report")

    if first_fire_frame is not None:
        st.error(f"🔥 Fire/Smoke first detected at frame {first_fire_frame} (~{first_fire_time:.1f} sec into video)")
        if "onset" in keyframes:
            st.image(cv2.cvtColor(keyframes["onset"], cv2.COLOR_BGR2RGB),
                      caption="Frame where fire/smoke was first detected", width=500)
    else:
        st.success("✅ No fire or smoke detected in this video.")

    total_frames = len(df)
    frames_with_fire = (df["fire_detections"] > 0).sum()
    frames_with_smoke = (df["smoke_detections"] > 0).sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Frames Analyzed", total_frames)
    col2.metric("Frames with Fire", frames_with_fire)
    col3.metric("Frames with Smoke", frames_with_smoke)

    st.subheader("Detection Confidence Over Time")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df["time_sec"], df["max_confidence"], color="red", marker="o", markersize=3)
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Max Confidence")
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

    st.subheader("Fire vs Smoke Detections Over Time")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(df["time_sec"], df["fire_detections"], label="Fire", color="orangered")
    ax2.plot(df["time_sec"], df["smoke_detections"], label="Smoke", color="gray")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Number of Detections")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    st.subheader("Overall Summary")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    categories = ["Fire Frames", "Smoke Frames", "Clear Frames"]
    values = [frames_with_fire, frames_with_smoke, total_frames - frames_with_fire - frames_with_smoke]
    ax3.bar(categories, values, color=["orangered", "gray", "green"])
    ax3.set_ylabel("Number of Frames")
    st.pyplot(fig3)

    with st.expander("View raw detection data"):
        st.dataframe(df)

    csv = df.to_csv(index=False)
    st.download_button("Download Report as CSV", csv, "fire_detection_report.csv", "text/csv")

def main_app():
    st.title("🔥 CCTV Fire & Smoke Detection Dashboard")
    st.write(f"Logged in as **{st.session_state.current_user}**")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    model = load_model()
    uploaded_video = st.file_uploader("Upload CCTV video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        if st.button("Run Detection"):
            with st.spinner("Processing video... this may take a minute"):
                results_data = process_video(video_path, model)
            display_results(results_data)

if st.session_state.logged_in:
    main_app()
else:
    login_page()