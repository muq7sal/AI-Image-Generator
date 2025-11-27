import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import json
import os
from datetime import datetime

# --- Streamlit page config ---
st.set_page_config(page_title="AI Text→Image Generator", layout="centered")
st.title("AI Text → Image Generator (HF API)")

# --- Hugging Face API token ---
HF_TOKEN = st.secrets["HF_API_TOKEN"]
HF_TOKEN = st.secrets.get("HF_API_TOKEN", "")
API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# --- Helper functions ---
SAMPLES_DIR = "sample_outputs"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def timestamp_folder():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def save_image_metadata(image: Image.Image, prompt: str, negative_prompt: str, folder: str):
    ensure_dir(folder)
    timestamp = datetime.now().strftime("%H%M%S")
    filename = f"image_{timestamp}.png"
    image.save(os.path.join(folder, filename))
    metadata = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "timestamp": datetime.now().isoformat(),
        "file": filename
    }
    with open(os.path.join(folder, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return filename

def hf_generate(prompt):
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=HEADERS, json=payload, stream=True)
    if response.status_code != 200:
        st.error(f"Error generating image: {response.status_code}, {response.text}")
        return None
    return Image.open(BytesIO(response.content))

# --- Prompt filtering ---
BANNED_KEYWORDS = {"porn", "rape", "child", "cp", "bestiality", "nsfw", "bomb"}

def is_prompt_allowed(prompt_text):
    lower = prompt_text.lower()
    for bad in BANNED_KEYWORDS:
        if bad in lower:
            return False, f"Prompt contains banned keyword: {bad}"
    return True, ""

# --- Streamlit UI ---
with st.form("generate_form"):
    prompt = st.text_area("Prompt", height=140, placeholder="e.g., A futuristic city at sunset, ultra-detailed, 4k")
    negative_prompt = st.text_input("Negative prompt (optional)", placeholder="e.g., low quality, blurry, watermark, text")
    submitted = st.form_submit_button("Generate")

if submitted:
    allowed, reason = is_prompt_allowed(prompt)
    if not allowed:
        st.error(reason)
    else:
        folder_name = os.path.join(SAMPLES_DIR, timestamp_folder())
        ensure_dir(folder_name)

        with st.spinner("Generating image... (may take a few seconds)"):
            img = hf_generate(prompt)
            if img:
                st.image(img, caption="Generated Image", use_column_width=True)
                # Save image + metadata
                filename = save_image_metadata(img, prompt, negative_prompt, folder_name)
                # Download button
                buf = BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                st.download_button("Download image", buf, file_name=filename, mime="image/png")
                st.success(f"Saved image & metadata to {folder_name}")




