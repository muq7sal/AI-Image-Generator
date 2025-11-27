# app.py
import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import json

st.set_page_config(page_title="AI Text → Image Generator", layout="centered")
st.title("AI Text → Image Generator (Stable Diffusion v1.5)")

# ---------- Utilities ----------
SAMPLES_DIR = "sample_outputs"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def timestamp_folder() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def save_images_with_metadata(images, prompt, negative_prompt, params, base_folder=SAMPLES_DIR):
    ensure_dir(base_folder)
    folder_name = timestamp_folder()
    folder = os.path.join(base_folder, folder_name)
    ensure_dir(folder)

    filenames = [f"image_{i+1}.png" for i in range(len(images))]
    for img, fname in zip(images, filenames):
        path = os.path.join(folder, fname)
        img.save(path)

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
    except Exception:
        font = ImageFont.load_default()
    text_w, text_h = draw.textsize(text, font=font)
    padding = 10
    x = img.size[0] - text_w - padding
    y = img.size[1] - text_h - padding
    draw.text((x, y), text, fill=(255,255,255,opacity), font=font)
    combined = Image.alpha_composite(img, txt_layer)
    return combined.convert("RGB")

# Very small blacklist
BANNED_KEYWORDS = {"porn", "rape", "child", "cp", "bestiality", "nsfw", "bomb"}
def is_prompt_allowed(prompt_text: str):
    lower = prompt_text.lower()
    for bad in BANNED_KEYWORDS:
        if bad in lower:
            return False, f"Prompt contains banned keyword: {bad}"
    return True, ""

# ---------- Model Loading ----------
@st.cache_resource(show_spinner=True)
def load_model(model_id="runwayml/stable-diffusion-v1-5"):
    HF_TOKEN = st.secrets["HF_API_TOKEN"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
        use_auth_token=HF_TOKEN
    )
    pipe = pipe.to(device)
    if device == "cuda":
        try:
            pipe.enable_attention_slicing()
        except:
            pass
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
    return pipe

pipe = load_model()

# ---------- Sidebar Settings ----------
st.sidebar.header("Defaults")
default_steps = st.sidebar.slider("Inference steps", 10, 100, 30)
default_guidance = st.sidebar.slider("Guidance scale", 1.0, 20.0, 7.5)
default_num = st.sidebar.slider("Default images per prompt", 1, 4, 1)

# ---------- Demo Prompt Selector ----------
DEMO_PROMPTS = {
    "Futuristic City": {
        "prompt": "A futuristic city at sunset, neon lights, flying cars, ultra-detailed, cinematic, 4K",
        "negative_prompt": "blurry, low quality, watermark, text, distorted, artifacts"
    },
    "Fantasy Forest": {
        "prompt": "Enchanted forest with glowing mushrooms, magical creatures, mystical fog, highly detailed, fantasy art",
        "negative_prompt": "dark, blurry, low quality, watermark, text"
    },
    "Robot Portrait": {
        "prompt": "Portrait of a humanoid robot in Van Gogh style, vivid colors, oil painting, detailed face, 4K",
        "negative_prompt": "blurry, low quality, text, watermark, distorted"
    }
}

demo_choice = st.selectbox("Choose Demo Prompt", ["Custom"] + list(DEMO_PROMPTS.keys()))
if demo_choice != "Custom":
    prompt_text = DEMO_PROMPTS[demo_choice]["prompt"]
    negative_text = DEMO_PROMPTS[demo_choice]["negative_prompt"]
else:
    prompt_text = ""
    negative_text = ""

# ---------- Main Form ----------
with st.form("generate_form"):
    prompt = st.text_area("Prompt", height=140, value=prompt_text)
    negative_prompt = st.text_input("Negative prompt (optional)", value=negative_text)
    num_images = st.slider("Number of images", 1, 4, default_num)
    steps = st.number_input("Inference steps", min_value=1, max_value=200, value=default_steps)
    guidance = st.number_input("Guidance scale", min_value=1.0, max_value=30.0, value=float(default_guidance))
    width = st.selectbox("Width", [512, 640, 768], index=0)
    height = st.selectbox("Height", [512, 640, 768], index=0)
    seed = st.number_input("Seed (0 for random)", value=0, step=1)
    watermark_toggle = st.checkbox("Apply 'AI GENERATED' watermark", value=True)
    submitted = st.form_submit_button("Generate")

# ---------- Generate Images ----------
if submitted:
    allowed, reason = is_prompt_allowed(prompt)
    if not allowed:
        st.error(f"Prompt blocked: {reason}")
    else:
        seed_val = None if int(seed) == 0 else int(seed)
        try:
            st.info("Generating images... (may take ~30-60s each)")
            generator = torch.Generator(device=pipe.device).manual_seed(seed_val) if seed_val else None
            images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt.strip() else None,
                num_images_per_prompt=num_images,
                guidance_scale=guidance,
                num_inference_steps=steps,
                height=height,
                width=width,
                generator=generator
            ).images

            # Display and download
            for i, im in enumerate(images):
                if watermark_toggle:
                    im = watermark_image(im, text="AI GENERATED")
                st.image(im, caption=f"Result {i+1}", use_column_width=True)
                buf = BytesIO()
                im.save(buf, format="PNG")
                st.download_button(
                    label=f"Download image {i+1} (PNG)",
                    data=buf.getvalue(),
                    file_name=f"image_{i+1}.png",
                    mime="image/png"
                )

            # Save images with metadata
            params = {
                "num_images": num_images,
                "guidance_scale": guidance,
                "steps": steps,
                "width": width,
                "height": height,
                "seed": seed_val
            }
            saved_folder = save_images_with_metadata(images, prompt, negative_prompt, params)
            st.success(f"Saved images & metadata to {saved_folder}")

        except Exception as e:
            st.error(f"Error generating image: {str(e)}")





