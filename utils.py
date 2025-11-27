# utils.py
import os
import json
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from typing import List

SAMPLES_DIR = "sample_outputs"
BANNED_KEYWORDS = {"porn", "rape", "child", "cp", "bestiality", "nsfw", "bomb"}

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def timestamp_folder() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def save_images_with_metadata(images: List[Image.Image], prompt: str, negative_prompt: str, params: dict, base_folder: str = SAMPLES_DIR):
    ensure_dir(base_folder)
    folder_name = timestamp_folder()
    folder = os.path.join(base_folder, folder_name)
    ensure_dir(folder)

    filenames = []
    for i, img in enumerate(images):
        fname = f"image_{i+1}.png"
        img.save(os.path.join(folder, fname))
        filenames.append(fname)

    meta = {"prompt": prompt, "negative_prompt": negative_prompt, "params": params, "timestamp": datetime.now().isoformat(), "files": filenames}
    with open(os.path.join(folder, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return folder

def watermark_image(pil_img: Image.Image, text="AI GENERATED", opacity=150) -> Image.Image:
    img = pil_img.convert("RGBA")
    txt_layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_layer)
    fontsize = max(12, img.size[0] // 30)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", fontsize)
    except Exception:
        font = ImageFont.load_default()
    text_w, text_h = draw.textsize(text, font=font)
    x = img.size[0] - text_w - 10
    y = img.size[1] - text_h - 10
    draw.text((x, y), text, fill=(255,255,255,opacity), font=font)
    combined = Image.alpha_composite(img, txt_layer)
    return combined.convert("RGB")

def is_prompt_allowed(prompt_text: str) -> (bool, str):
    lower = prompt_text.lower()
    for bad in BANNED_KEYWORDS:
        if bad in lower:
            return False, f"Prompt contains banned keyword: {bad}"
    return True, ""
