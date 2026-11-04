"""Streamlit interface for brain-tumor prediction and Grad-CAM visualization."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from src.gradcam import explain_image


st.set_page_config(page_title="Brain Tumor XAI", page_icon="🧠", layout="wide")
st.title("Brain Tumor Classification with Explainable AI")
st.write(
    "Upload an MRI scan to obtain a four-class prediction and a Grad-CAM "
    "visualization of the regions used by the model."
)

checkpoint_path = Path(
    st.sidebar.text_input("Model checkpoint", "artifacts/best_model.pt")
)
device_options = ("auto", "cpu", "cuda") if torch.cuda.is_available() else ("auto", "cpu")
device = st.sidebar.selectbox("Inference device", device_options)
uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=("jpg", "jpeg", "png", "bmp", "tif", "tiff"),
)

if uploaded_file is not None:
    original = Image.open(uploaded_file).convert("RGB")
    st.subheader("Input scan")
    st.image(original, use_container_width=True)

    if not checkpoint_path.is_file():
        st.error(f"Checkpoint not found: {checkpoint_path}")
        st.stop()

    if st.button("Analyze scan", type="primary"):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_directory_path = Path(temporary_directory)
            image_path = temporary_directory_path / Path(uploaded_file.name).name
            overlay_path = temporary_directory_path / "gradcam_overlay.png"
            image_path.write_bytes(uploaded_file.getvalue())
            selected_device = None if device == "auto" else device
            with st.spinner("Running model inference and Grad-CAM..."):
                result = explain_image(
                    checkpoint=checkpoint_path,
                    image_path=image_path,
                    output_path=overlay_path,
                    device=selected_device,
                )
            st.subheader("Prediction")
            st.metric(
                label="Predicted class",
                value=str(result["class"]).replace("_", " ").title(),
                delta=f"{float(result['confidence']) * 100:.2f}% confidence",
            )
            st.subheader("Grad-CAM explanation")
            st.image(
                Image.open(overlay_path),
                caption="Red/yellow regions contributed most strongly to the prediction.",
                use_container_width=True,
            )
else:
    st.info("Choose an MRI image above to begin.")
