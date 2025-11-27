# app.py
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
pipeline = load_model(device=device)

# User inputs
prompt = st.text_area("Enter your prompt:", height=100)
num_images = st.slider("Number of images", 1, 5, 1)
guidance_scale = st.slider("Guidance scale (creativity control)", 1.0, 20.0, 7.5)

# Image generation
if st.button("Generate Images"):
    if not prompt:
        st.warning("Please enter a prompt!")
    else:
        with st.spinner("Generating image(s)... This may take some time."):
            images = generate_images(pipeline, prompt, num_images, guidance_scale)

        st.success("Images Generated Successfully!")

        # Save images and metadata
        os.makedirs("images", exist_ok=True)
        metadata_file = "images/metadata.csv"
        with open(metadata_file, "a", newline="") as f:
            writer = csv.writer(f)
            for idx, img in enumerate(images):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"images/{timestamp}_{idx}.png"
                img.save(filename)
                writer.writerow([timestamp, prompt, num_images, guidance_scale, filename])
                st.image(img, caption=f"Generated Image {idx+1}", use_column_width=True)
                st.download_button(
                    "Download Image",
                    data=img.tobytes(),
                    file_name=f"{timestamp}_{idx}.png",
                    mime="image/png"
                )
