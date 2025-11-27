# utils.py
import os
import json
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

SAMPLES_DIR = "sample_outputs"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def timestamp_folder():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def save_images_with_metadata(images, prompt, negative_prompt, params: dict, base_folder=SAMPLES_DIR, filenames=None):
    ensure_dir(base_folder)
    folder = os.path.join(base_folder, timestamp_folder())
    ensure_dir(folder)
    meta = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "params": params,
        "timestamp": datetime.now().isoformat()
    }
    if filenames is None:
        filenames = [f"image_{i+1}.png" for i in range(len(images))]
    for img, fname in zip(images, filenames):
        path = os.path.join(folder, fname)
        img.save(path)
    with open(os.path.join(folder, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return folder

# Simple watermark (bottom-right) - uses a basic PIL font
def watermark_image(pil_img: Image.Image, text="AI GENERATED", opacity=128):
    img = pil_img.convert("RGBA")
    txt_layer = Image.new("RGBA", img.size, (255,255,255,0))
    draw = ImageDraw.Draw(txt_layer)
    # choose a font size relative to image
    fontsize = max(12, img.size[0] // 20)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", fontsize)
    except Exception:
        font = ImageFont.load_default()
    text_w, text_h = draw.textsize(text, font=font)
    padding = 10
    x = img.size[0] - text_w - padding
    y = img.size[1] - text_h - padding
    draw.text((x, y), text, fill=(255,255,255,opacity), font=font)
    combined = Image.alpha_composite(img, txt_layer)
    return combined.convert("RGB")

# Very simple content filter -- RULES: block if banned words
BANNED_KEYWORDS = {"porn", "rape", "child", "cp", "bestiality", "nsfw", "bomb"}

def is_prompt_allowed(prompt_text: str) -> (bool, str):
    lower = prompt_text.lower()
    for bad in BANNED_KEYWORDS:
        if bad in lower:
            return False, f"Prompt contains banned keyword: {bad}"
    return True, ""
