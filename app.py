# app.py
import torch
import streamlit as st
from engine import load_model, generate_images
from PIL import Image
import os
from datetime import datetime
import csv

# Page setup
st.set_page_config(page_title="AI-Powered Image Generator", layout="wide")
st.title("AI-Powered Text-to-Image Generator")

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
st.info(f"Using device: {device.upper()}")

@st.cache_resource(show_spinner=True)
def get_pipeline():
    return load_model(device=device)

pipeline = get_pipeline()

# User inputs
prompt = st.text_area("Enter your prompt:", height=100)
num_images = st.slider("Number of images", 1, 2, 1)  # Keep 1–2 for Cloud speed
guidance_scale = st.slider("Guidance scale (creativity control)", 1.0, 20.0, 7.5)

# Generate Images
if st.button("Generate Images"):
    if not prompt.strip():
        st.warning("Please enter a prompt!")
    else:
        with st.spinner("Generating image(s)... This may take a minute on CPU."):
            images = generate_images(pipeline, prompt, num_images, guidance_scale)
        st.success("Images Generated!")

        # Save images & metadata
        os.makedirs("images", exist_ok=True)
        metadata_file = "images/metadata.csv"
        if not os.path.exists(metadata_file):
            with open(metadata_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Prompt", "Num_Images", "Guidance_Scale", "Filename"])

        for idx, img in enumerate(images):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"images/{timestamp}_{idx}.png"
            img.save(filename)

            with open(metadata_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, prompt, num_images, guidance_scale, filename])

            st.image(img, caption=f"Generated Image {idx+1}", use_column_width=True)
            st.download_button(
                label="Download Image",
                data=img.tobytes(),
                file_name=f"{timestamp}_{idx}.png",
                mime="image/png"
            )
