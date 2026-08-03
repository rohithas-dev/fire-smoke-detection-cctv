import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import tempfile
import os
import subprocess

# ---------- CONFIG ----------
MODEL_PATH = "fire-and-smoke-detection-yolov8/weights/best.pt"
VALID_USERS = {
    "admin": "fire123",
    "teacher": "demo2024",
    "guest": "guest123"
}
FRAME_SKIP = 5
CONF_THRESHOLD = 0.4
EMERGENCY_FIRE_NUMBER = "101"       # India Fire Services
EMERGENCY_GENERAL_NUMBER = "112"   # India National Emergency Number
MAX_SCRUB_FRAMES = 24              # cap stored frames for the scrub viewer (memory safety)
# -----------------------------

st.set_page_config(page_title="Fire & Smoke CCTV Detection", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


def login_page():
    st.markdown("""
        <style>
        .stApp {
            background: #000000;
        }

        #bonfire-video {
            width: 100%;
            max-width: 450px;
            height: 550px;
            object-fit: cover;
            border-radius: 12px;
            transition: opacity 0.6s ease;
        }

        #bonfire-video.out {
            opacity: 0.15;
        }

        .login-title {
            text-align: center;
            color: #ffffff;
            font-size: 30px;
            font-weight: 700;
            margin-bottom: 5px;
            text-shadow: 0 0 10px rgba(255, 87, 34, 0.5);
        }

        .login-subtitle {
            text-align: center;
            color: #a0a0c0;
            font-size: 14px;
            margin-bottom: 20px;
        }

        div[data-testid="stTextInput"] input {
            background-color: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 87, 34, 0.3);
            border-radius: 8px;
            color: black;
        }

        div[data-testid="stTextInput"] input:focus {
            border: 1px solid #ff5722;
            box-shadow: 0 0 10px rgba(255, 87, 34, 0.4);
        }

        .stButton button {
            background: linear-gradient(135deg, #ff5722, #ff9800);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 30px;
            font-weight: 600;
            width: 100%;
        }

        label { color: #d0d0e0 !important; }
        </style>
    """, unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1.3])

    with left_col:
        st.markdown("""
            <video id="bonfire-video" autoplay loop muted playsinline>
                <source src="https://raw.githubusercontent.com/rohithas-dev/fire-smoke-detection-cctv/main/fire.mp4" type="video/mp4">
            </video>
        """, unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="login-title">CCTV Fire & Smoke Detection</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-subtitle">Real-time monitoring powered by YOLOv8</div>', unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", placeholder="Enter password", key="password_field")

        if st.button("Login"):
            if username in VALID_USERS and VALID_USERS[username] == password:
                st.session_state.logged_in = True
                st.session_state.current_user = username
                st.rerun()
            else:
                st.error("Invalid username or password")

        st.markdown(
            "<p style='text-align:center; color:#808090; font-size:12px; margin-top:15px;'>Demo credentials: admin / fire123</p>",
            unsafe_allow_html=True
        )

    st.components.v1.html("""
        <script>
        const doc = window.parent.document;
        function attachListener() {
            const passwordInputs = doc.querySelectorAll('input[type="password"]');
            const video = doc.getElementById('bonfire-video');
            if (passwordInputs.length > 0 && video) {
                passwordInputs[0].addEventListener('focus', () => {
                    video.classList.add('out');
                    video.pause();
                });
                passwordInputs[0].addEventListener('blur', () => {
                    video.classList.remove('out');
                    video.play();
                });
            } else {
                setTimeout(attachListener, 300);
            }
        }
        attachListener();
        </script>
    """, height=0)


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def draw_detections(frame, boxes, names):
    """Draw persisted bounding boxes on a frame (used for frames between YOLO samples too)."""
    annotated = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box["xyxy"])
        color = (0, 0, 255) if box["label"] == "fire" else (255, 165, 0)
        label = f'{box["label"]} {box["conf"]:.2f}'
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return annotated


def convert_to_h264(input_path):
    """Convert OpenCV generated mp4 (mp4v codec) to H.264 format so modern browsers can display it in HTML5 video element."""
    h264_path = tempfile.NamedTemporaryFile(delete=False, suffix="_h264.mp4").name
    
    # 1. Try standard ffmpeg system binary
    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-acodec", "aac", h264_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.exists(h264_path) and os.path.getsize(h264_path) > 0:
            return h264_path
    except Exception:
        pass

    # 2. Try imageio_ffmpeg if installed in environment
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y", "-i", input_path,
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-acodec", "aac", h264_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.exists(h264_path) and os.path.getsize(h264_path) > 0:
            return h264_path
    except Exception:
        pass

    return input_path


def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idx = 0
    records = []
    first_fire_frame = None
    first_fire_time = None
    first_smoke_frame = None
    first_smoke_time = None
    keyframes = {}
    scrub_frames = []
    last_boxes = []

    raw_annotated_path = tempfile.NamedTemporaryFile(delete=False, suffix="_raw.mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_annotated_path, fourcc, fps, (width, height))

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

            last_boxes = [
                {
                    "xyxy": boxes.xyxy[i].tolist(),
                    "label": model.names[int(boxes.cls[i])],
                    "conf": float(boxes.conf[i])
                }
                for i in range(len(boxes))
            ]

            records.append({
                "frame": frame_idx,
                "time_sec": round(timestamp, 2),
                "fire_detections": fire_count,
                "smoke_detections": smoke_count,
                "max_confidence": max_conf
            })

            if fire_count > 0 and first_fire_frame is None:
                first_fire_frame = frame_idx
                first_fire_time = timestamp

            if smoke_count > 0 and first_smoke_frame is None:
                first_smoke_frame = frame_idx
                first_smoke_time = timestamp

            if (fire_count > 0 or smoke_count > 0) and len(scrub_frames) < MAX_SCRUB_FRAMES:
                annotated_preview = draw_detections(frame, last_boxes, model.names)
                scrub_frames.append({
                    "frame": frame_idx,
                    "time_sec": round(timestamp, 2),
                    "image": cv2.cvtColor(annotated_preview, cv2.COLOR_BGR2RGB)
                })
                if first_fire_frame == frame_idx or "onset" not in keyframes:
                    keyframes["onset"] = cv2.cvtColor(annotated_preview, cv2.COLOR_BGR2RGB)

        # Draw the last known boxes on every frame so tracking looks continuous, not just every Nth frame
        annotated_frame = draw_detections(frame, last_boxes, model.names)
        writer.write(annotated_frame)

        frame_idx += 1

    cap.release()
    writer.release()

    # Transcode video to H.264 so standard web browsers can render it smoothly in st.video()
    playable_video_path = convert_to_h264(raw_annotated_path)

    df = pd.DataFrame(records)
    return {
        "df": df,
        "first_fire_frame": first_fire_frame,
        "first_fire_time": first_fire_time,
        "first_smoke_frame": first_smoke_frame,
        "first_smoke_time": first_smoke_time,
        "keyframes": keyframes,
        "scrub_frames": scrub_frames,
        "annotated_video_path": playable_video_path,
    }


def build_narrative(first_fire_frame, first_fire_time, first_smoke_frame, first_smoke_time, df):
    if first_fire_frame is None and first_smoke_frame is None:
        return ("No fire or smoke activity was detected in the footage. "
                "The scene remained clear throughout the recording.")

    parts = []
    if first_smoke_frame is not None and first_fire_frame is not None:
        if first_smoke_time < first_fire_time:
            gap = first_fire_time - first_smoke_time
            parts.append(
                f"Smoke was first observed at approximately {first_smoke_time:.1f} seconds into the "
                f"footage, roughly {gap:.1f} seconds before visible flame appeared at "
                f"{first_fire_time:.1f} seconds. This pattern is consistent with a smoldering onset "
                f"that progressed into open flame rather than a sudden flash fire."
            )
        else:
            parts.append(
                f"Flame was detected at {first_fire_time:.1f} seconds, with smoke becoming visible "
                f"shortly after at {first_smoke_time:.1f} seconds — consistent with a fast-developing "
                f"ignition rather than a slow smolder."
            )
    elif first_fire_frame is not None:
        parts.append(
            f"Flame was detected starting at approximately {first_fire_time:.1f} seconds into the "
            f"footage, with no distinct smoke phase captured beforehand."
        )
    elif first_smoke_frame is not None:
        parts.append(
            f"Smoke was detected starting at approximately {first_smoke_time:.1f} seconds into the "
            f"footage. No open flame was confirmed by the model in this recording."
        )

    if not df.empty:
        peak_row = df.loc[df["max_confidence"].idxmax()]
        parts.append(
            f"Detection confidence peaked at {peak_row['max_confidence']:.2f} around "
            f"{peak_row['time_sec']:.1f} seconds, marking the clearest visual evidence of fire or "
            f"smoke captured in the footage."
        )

    return " ".join(parts)


def style_matplotlib_figure(fig, ax, title, xlabel, ylabel):
    """Apply rich dark theme and sleek styling to matplotlib figures."""
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b26")
    
    # Title and axis labels
    ax.set_title(title, color="#ffffff", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color="#a0a0c0", fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, color="#a0a0c0", fontsize=10, labelpad=8)
    
    # Ticks & Grid
    ax.tick_params(colors="#a0a0c0", labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.25, color="#505568")
    
    # Spines border color
    for spine in ax.spines.values():
        spine.set_color("#2d3345")
        spine.set_linewidth(1.2)


def display_results(data):
    df = data["df"]
    first_fire_frame = data["first_fire_frame"]
    first_fire_time = data["first_fire_time"]
    first_smoke_frame = data["first_smoke_frame"]
    first_smoke_time = data["first_smoke_time"]
    scrub_frames = data["scrub_frames"]
    annotated_video_path = data["annotated_video_path"]

    st.header("Detection Report")

    if first_fire_frame is not None:
        st.error(f"Fire first detected at frame {first_fire_frame} (~{first_fire_time:.1f} sec into video)")
    elif first_smoke_frame is not None:
        st.warning(f"Smoke first detected at frame {first_smoke_frame} (~{first_smoke_time:.1f} sec into video)")
    else:
        st.success("No fire or smoke detected in this video.")

    total_frames = len(df)
    frames_with_fire = int((df["fire_detections"] > 0).sum()) if not df.empty else 0
    frames_with_smoke = int((df["smoke_detections"] > 0).sum()) if not df.empty else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Frames Analyzed", total_frames)
    col2.metric("Frames with Fire", frames_with_fire)
    col3.metric("Frames with Smoke", frames_with_smoke)

    st.subheader("Live Detection Tracking")
    # Stream bytes to guarantee HTML5 video playback in Streamlit
    try:
        with open(annotated_video_path, "rb") as video_file:
            video_bytes = video_file.read()
            st.video(video_bytes, format="video/mp4")
    except Exception:
        st.video(annotated_video_path)

    st.caption("Bounding boxes track fire/smoke detections throughout the footage.")
    with open(annotated_video_path, "rb") as f:
        st.download_button("Download Annotated Video", f, "annotated_detection.mp4", "video/mp4")

    if scrub_frames:
        st.subheader("Scrub Through Detections")
        idx = st.slider("Detection #", 0, len(scrub_frames) - 1, 0)
        chosen = scrub_frames[idx]
        st.image(chosen["image"], caption=f"Frame {chosen['frame']} (~{chosen['time_sec']:.1f}s)", width=500)

    # Vibrant Colored Graph 1: Detection Confidence Over Time
    st.subheader("Detection Confidence Over Time")
    fig1, ax1 = plt.subplots(figsize=(10, 4.2))
    style_matplotlib_figure(fig1, ax1, "Detection Confidence Progression", "Time (seconds)", "Max Confidence Score")
    
    if not df.empty:
        # Gradient area fill under curve with vibrant fiery red/orange line
        ax1.plot(df["time_sec"], df["max_confidence"], color="#ff4500", linewidth=2.5, label="Max Confidence", marker="o", markersize=4, mfc="#ffaa00", mec="#ff4500")
        ax1.fill_between(df["time_sec"], df["max_confidence"], color="#ff4500", alpha=0.22)
        ax1.set_ylim(-0.05, 1.05)
    
    st.pyplot(fig1)

    # Vibrant Colored Graph 2: Fire vs Smoke Detections Over Time
    st.subheader("Fire vs Smoke Detections Over Time")
    fig2, ax2 = plt.subplots(figsize=(10, 4.2))
    style_matplotlib_figure(fig2, ax2, "Fire vs Smoke Frequency Over Time", "Time (seconds)", "Number of Bounding Boxes")
    
    if not df.empty:
        # Fire line (Vibrant Flame Orange/Red)
        ax2.plot(df["time_sec"], df["fire_detections"], label="Fire", color="#ff5722", linewidth=2.5, marker="s", markersize=4, mfc="#ff8a65")
        ax2.fill_between(df["time_sec"], df["fire_detections"], color="#ff5722", alpha=0.18)
        
        # Smoke line (Vibrant Cyan / Smoke Blue)
        ax2.plot(df["time_sec"], df["smoke_detections"], label="Smoke", color="#00e5ff", linewidth=2.5, marker="o", markersize=4, mfc="#80d8ff")
        ax2.fill_between(df["time_sec"], df["smoke_detections"], color="#00e5ff", alpha=0.15)
        
        legend = ax2.legend(facecolor="#1e2433", edgecolor="#3a4259", labelcolor="#ffffff", fontsize=10)
        legend.get_frame().set_alpha(0.9)
    
    st.pyplot(fig2)

    with st.expander("View raw detection data"):
        st.dataframe(df)

    csv = df.to_csv(index=False)
    st.download_button("Download Report as CSV", csv, "fire_detection_report.csv", "text/csv")

    pdf_bytes = generate_pdf_report(df, first_fire_frame, first_fire_time,
                                     first_smoke_frame, first_smoke_time,
                                     total_frames, frames_with_fire, frames_with_smoke,
                                     fig1, fig2)
    st.download_button("Download Report as PDF", pdf_bytes, "fire_detection_report.pdf", "application/pdf")


def generate_pdf_report(df, first_fire_frame, first_fire_time, first_smoke_frame, first_smoke_time,
                         total_frames, frames_with_fire, frames_with_smoke, fig1, fig2):
    import io
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.units import inch

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.6*inch, bottomMargin=0.6*inch)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("CCTV Fire and Smoke Detection Report", styles['Title']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Incident Summary", styles['Heading2']))
    story.append(Paragraph(
        build_narrative(first_fire_frame, first_fire_time, first_smoke_frame, first_smoke_time, df),
        styles['Normal']
    ))
    story.append(Spacer(1, 16))

    summary_data = [
        ["Metric", "Value"],
        ["Frames Analyzed", str(total_frames)],
        ["Frames with Fire", str(frames_with_fire)],
        ["Frames with Smoke", str(frames_with_smoke)],
    ]
    table = Table(summary_data, colWidths=[200, 200])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#ff5722")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    for fig, caption in [(fig1, "Detection Confidence Over Time"),
                          (fig2, "Fire vs Smoke Detections Over Time")]:
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        img_buffer.seek(0)
        story.append(Paragraph(caption, styles['Heading2']))
        story.append(Image(img_buffer, width=6*inch, height=2.5*inch))
        story.append(Spacer(1, 14))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Emergency Contacts", styles['Heading2']))
    story.append(Paragraph(
        f"If this event is active, evacuate the area immediately and contact the Fire Services at "
        f"{EMERGENCY_FIRE_NUMBER} or the National Emergency Number at {EMERGENCY_GENERAL_NUMBER}.",
        styles['Normal']
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def main_app():
    header_left, header_right = st.columns([6, 1])
    with header_left:
        st.title("CCTV Fire & Smoke Detection Dashboard")
        st.write(f"Logged in as **{st.session_state.current_user}**")
    with header_right:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Logout", use_container_width=True):
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