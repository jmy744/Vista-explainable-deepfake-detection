"""Interview presentation of the original CNN–BiLSTM–GAT checkpoint."""
import hashlib
import json
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch

from model import load_model
from video_processor import VideoProcessor, NoFaceDetected
from explainability import compute_gradcam, compute_temporal_attention, compute_gat_attention
from visualizer import create_explainability_figure

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Deepfake Research Demo", page_icon="◈", layout="wide")
st.markdown('''<style>
.block-container{max-width:1180px;padding-top:2.5rem}
h1,h2,h3{letter-spacing:-.03em} .eyebrow{color:#64748b;font-size:.8rem;letter-spacing:.14em;text-transform:uppercase}
[data-testid="stMetric"]{background:#f1f5f9;border-radius:12px;padding:16px;color:#0f172a}
</style>''', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">Final-year project · Research prototype</div>', unsafe_allow_html=True)
st.title("Explainable video deepfake detection")
st.write("Explore a prediction, inspect the visual evidence, and see how CNN, BiLSTM and GAT work together.")

@st.cache_resource
def resources():
    # CPU keeps this interview build portable and avoids cuDNN eval-mode RNN backward.
    torch.set_num_threads(2)
    model, meta = load_model(ROOT / "proposed_best.pth", device="cpu", verbose=False)
    return model, meta, threading.Lock()

try:
    model, meta, lock = resources()
except Exception as exc:
    st.error(f"Model could not load: {exc}")
    st.info("Keep proposed_best.pth beside app.py and install the supplied requirements.")
    st.stop()

with st.sidebar:
    
    st.write("ResNet34 + BiLSTM + landmark GAT")
    st.caption("20 sampled frames · 224 × 224 face crops · 68 landmark nodes · threshold 0.50 · CPU")
    st.success("Checkpoint loaded")
    st.divider()

demo, architecture = st.tabs(["Analyze a video", "Architecture"])

with architecture:
    st.header("Two streams, one prediction")
    st.graphviz_chart('''digraph G {
      rankdir=TB; node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Helvetica"];
      video [label="Video · 20 sampled frames"];
      crop [label="Face crops · 224 × 224"];
      points [label="68 landmarks · x, y, z"];
      cnn [label="ResNet34 CNN · visual features"];
      lstm [label="2-layer BiLSTM · sequence features"];
      gat [label="2-layer GAT · facial graph features"];
      mean [label="Mean across frames"];
      fusion [label="Concatenate 512 + 128 features"];
      result [label="Classifier · real / fake scores"];
      video -> crop -> cnn -> lstm -> fusion;
      video -> points -> gat -> mean -> fusion;
      fusion -> result;
    }''')
    st.markdown("""
    **CNN:** extracts visual patterns from each face crop.  
    **BiLSTM:** processes the sequence of CNN features.  
    **GAT:** combines connected facial landmark features; graph outputs are averaged across frames.  
    **Fusion:** combines both streams before classification.

    The heatmap comes from CNN Grad-CAM. The frame curve measures sensitivity to replacing
    a visual feature. The graph panel exposes GAT attention coefficients.
    """)
    
with demo:
    st.subheader("1 · Choose a face video")
    st.caption("Use an original video with one clearly visible face. A screen recording of this app is not a suitable evaluation clip.")
    uploaded = st.file_uploader("Upload MP4, MOV or AVI", type=["mp4", "mov", "avi", "mkv"])
    if uploaded is not None:
        data = uploaded.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        st.video(data)
        if st.session_state.get("video_digest") != digest:
            st.session_state.pop("analysis", None)
            st.session_state["video_digest"] = digest
        if st.button("Analyze video", type="primary"):
            st.session_state.pop("analysis", None)
            path = None
            processor = None
            try:
                started = time.perf_counter()
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as f:
                    f.write(data)
                    path = Path(f.name)
                with st.spinner("Sampling faces, running the model and computing explanations…"):
                    processor = VideoProcessor(num_frames=20)
                    video, landmarks, crops, info = processor.process_video(str(path))
                    with lock:
                        cams, target, probs = compute_gradcam(model, video, landmarks, device="cpu")
                        temporal, _ = compute_temporal_attention(model, video, landmarks, target_class=target, device="cpu")
                        peak = int(np.argmax(temporal))
                        edges, stats = compute_gat_attention(model, landmarks, frame_idx=peak, device="cpu")
                    if not np.isfinite(probs).all() or not np.isfinite(cams).all():
                        raise RuntimeError("The model returned non-finite values.")
                    st.session_state.analysis = dict(crops=crops, cams=cams, temporal=temporal, peak=peak,
                        edges=edges, stats=stats, probs=probs, info=info,
                        seconds=time.perf_counter()-started, filename=uploaded.name)
            except NoFaceDetected as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"Analysis could not complete: {exc}")
            finally:
                if processor is not None:
                    processor.face_mesh.close()
                if path is not None:
                    path.unlink(missing_ok=True)

    r = st.session_state.get("analysis") if uploaded is not None else None
    if r:
        st.subheader("2 · Model prediction")
        fake = float(r["probs"][1])
        label = "Fake" if fake >= 0.5 else "Real"
        a,b,c = st.columns(3)
        a.metric("Predicted class", label)
        b.metric("Fake-class model score", f"{fake:.3f}")
        c.metric("Faces found at sampled positions", f"{r['info']['face_detected_frames']}/20")
        st.caption(f"Threshold 0.50 · analysis took {r['seconds']:.1f}s · class mapping retained from uploaded demo: real=0, fake=1.")
        missed = 20 - r["info"]["face_detected_frames"]
        if missed:
            st.warning(f"Face detection failed at {missed} sampled positions. The original preprocessing fills missing inputs by repeating crops. Interpret this result cautiously.")
        if r["info"]["total_frames"] < 20:
            st.warning("This short video has fewer than 20 frames; sampling reuses frames.")
        st.subheader("3 · Inspect the explanation")
        frame = st.slider("Sample position (not a timestamp)", 0, 19, r["peak"])
        if not np.any(r["temporal"]):
            st.info("No score change was measured by the visual feature-replacement check.")
        edges, stats = r["edges"], r["stats"]
        figure = create_explainability_figure(r["crops"], r["cams"], r["temporal"], edges,
            stats, int(fake>=0.5), r["probs"], peak_frame=frame, show_structural=False)
        st.image(figure, use_column_width=True)
        st.caption("Heatmaps are normalized separately for each frame; intensity is not directly comparable between frames. The curve measures absolute changes in the predicted-class score after visual feature replacement; graph features stay fixed.")
        with st.expander(f"GAT attention diagnostic · sample position {r['peak']}"):
            st.write("These coefficients weight landmark connections. They are class-agnostic and do not identify verified manipulated regions.")
            st.dataframe(sorted(r["edges"], key=lambda e: abs(e["deviation"]), reverse=True), hide_index=True, use_container_width=True)
            st.caption(f"Maximum deviation from equal-neighbor weighting: {r['stats']['max_abs_deviation']:.6f}. This statistic is descriptive, not an explanation-quality score.")
        st.download_button("Download explanation figure", figure, "explanation.png", "image/png")
        report = {"filename":r["filename"],"prediction":label,"fake_score":fake,"threshold":0.5,
            "video_sha256":st.session_state.video_digest,"sampling":r["info"],"seconds":r["seconds"],
            "visual_frame_sensitivity":r["temporal"].tolist(),"gat_diagnostic":r["stats"],
            "scope":"Research prototype output, not verified authenticity or manipulation localization"}
        st.download_button("Download analysis record", json.dumps(report, indent=2), "analysis.json", "application/json")
