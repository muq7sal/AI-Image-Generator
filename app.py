# app.py
import streamlit as st
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import os
import json

st.set_page_config(page_title="AI Text → Image Generator", layout="centered")
st.title("AI Text → Image Generator (Hugging Face Router API)")

# ---- Secrets ----
HF_TOKEN = st.secrets["HF_API_TOKEN"]
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
API_URL = "https://router.huggingface.co/models/runwayml/stable-diffusion-v1-5"

# ---- Utils ----
SAMPLES_DIR = "sample_outputs"
BANNED_KEYWORDS = {"porn", "rape", "child", "cp", "bestiality", "nsfw", "bomb"}

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def timestamp_folder() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def save_images_with_metadata(images, prompt, negative_prompt, params):
    ensure_dir(SAMPLES_DIR)
    folder_name = timestamp_folder()
    folder = os.path.join(SAMPLES_DIR, folder_name)
    ensure_dir(folder)

    filenames = []
    for i, img in enumerate(images):
        fname = f"image_{i+1}.png"
        img.save(os.path.join(folder, fname))
        filenames.append(fname)

    meta = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "params": params,
        "timestamp": datetime.now().isoformat(),
        "file_list": filenames
    }
    with open(os.path.join(folder, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return folder

def watermark_image(pil_img: Image.Image, text: str = "AI GENERATED", opacity: int = 150) -> Image.Image:
    img = pil_img.convert("RGBA")
    txt_layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_layer)
    fontsize = max(12, img.size[0] // 30)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", fontsize)
    except:
        font = ImageFont.load_default()
    text_w, text_h = draw.textsize(text, font=font)
    padding = 10
    x = img.size[0] - text_w - padding
    y = img.size[1] - text_h - padding
    draw.text((x, y), text, fill=(255, 255, 255, opacity), font=font)
    combined = Image.alpha_composite(img, txt_layer)
    return combined.convert("RGB")

def is_prompt_allowed(prompt_text: str):
    lower = prompt_text.lower()
    for bad in BANNED_KEYWORDS:
        if bad in lower:
            return False, f"Prompt contains banned keyword: {bad}"
    return True, ""

# ---- Sidebar ----
st.sidebar.header("Defaults")
default_num = st.sidebar.slider("Default images per prompt", 1, 4, 1)

# ---- Main form ----
with st.form("generate_form"):
    prompt = st.text_area("Prompt", height=140, placeholder="A futuristic city at sunset, ultra-detailed, 4K")
    negative_prompt = st.text_input("Negative prompt (optional)", value="blurry, low quality, watermark, text, distorted, artifacts")
    num_images = st.slider("Number of images", 1, 4, default_num)
    watermark_toggle = st.checkbox("Apply 'AI GENERATED' watermark", value=True)
    submitted = st.form_submit_button("Generate")

# ---- Generate images ----
if submitted:
    allowed, reason = is_prompt_allowed(prompt)
    if not allowed:
        st.error(f"Prompt blocked: {reason}")
    else:
        images = []
        status = st.empty()
        for i in range(num_images):
            status.info(f"Generating image {i+1}/{num_images}... (may take ~30-60s each)")

            payload = {
                "inputs": prompt,
                "parameters": {
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": 30,
                    "guidance_scale": 7.5,
                    "width": 512,
                    "height": 512
                },
                "options": {"wait_for_model": True}
            }

            response = requests.post(API_URL, headers=HEADERS, json=payload, stream=True)
            if response.status_code != 200:
                st.error(f"Error generating image: {response.status_code}\n{response.text}")
                break

            img = Image.open(BytesIO(response.content))
            if watermark_toggle:
                img = watermark_image(img)
            st.image(img, caption=f"Result {i+1}", use_column_width=True)
            images.append(img)

        if images:
            params = {"num_images": num_images, "negative_prompt": negative_prompt}
            folder = save_images_with_metadata(images, prompt, negative_prompt, params)
            status.success(f"Saved {len(images)} images & metadata to {folder}")





