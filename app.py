# app.py
import streamlit as st
from engine import TextToImageEngine
from utils import save_images_with_metadata, watermark_image, is_prompt_allowed
from io import BytesIO

st.set_page_config(page_title="Text → Image Generator", layout="centered")
st.title("AI Text → Image Generator (Stable Diffusion v1.5)")

# Sidebar - model & hardware options
st.sidebar.header("Model & Hardware")
device_choice = st.sidebar.selectbox("Device", options=["auto", "cuda", "cpu"], index=0)
model_id = st.sidebar.text_input("Model ID", value="runwayml/stable-diffusion-v1-5")

st.sidebar.header("Defaults")
default_steps = st.sidebar.slider("Inference steps", 10, 100, 30)
default_guidance = st.sidebar.slider("Guidance scale", 1.0, 20.0, 7.5)
default_num = st.sidebar.slider("Default images per prompt", 1, 4, 1)

# Main form
with st.form("generate_form"):
    prompt = st.text_area("Prompt", height=140, placeholder="e.g., A futuristic city at sunset, ultra-detailed, 4k")
    negative_prompt = st.text_input("Negative prompt (optional)", placeholder="e.g., low quality, blurry, watermark, text")
    num_images = st.slider("Number of images", 1, 4, default_num)
    steps = st.number_input("Inference steps", min_value=1, max_value=200, value=default_steps)
    guidance = st.number_input("Guidance scale", min_value=1.0, max_value=30.0, value=float(default_guidance))
    width = st.selectbox("Width", [512, 640, 768], index=0)
    height = st.selectbox("Height", [512, 640, 768], index=0)
    seed = st.number_input("Seed (0 for random)", value=0, step=1)
    watermark_toggle = st.checkbox("Apply 'AI GENERATED' watermark", value=True)
    submitted = st.form_submit_button("Generate")

if submitted:
    allowed, reason = is_prompt_allowed(prompt)
    if not allowed:
        st.error(f"Prompt blocked: {reason}")
    else:
        # choose device
        device_param = None
        if device_choice == "auto":
            device_param = None
        else:
            device_param = device_choice

        # load engine into session state (cache)
        if "engine" not in st.session_state or st.session_state.get("engine_model") != model_id:
            with st.spinner("Loading model (may take a minute)..."):
                st.session_state["engine"] = TextToImageEngine(model_id=model_id, device=device_param)
                st.session_state["engine_model"] = model_id

        engine = st.session_state["engine"]

        status = st.empty()
        status.info("Generating...")

        seed_val = None if int(seed) == 0 else int(seed)
        images = engine.generate(
            prompt=prompt,
            negative_prompt=(negative_prompt if negative_prompt.strip() else None),
            num_images=num_images,
            guidance_scale=guidance,
            num_inference_steps=steps,
            height=height,
            width=width,
            seed=seed_val
        )

        # Apply watermark, display and provide downloads
        filenames = []
        for i, im in enumerate(images):
            if watermark_toggle:
                im = watermark_image(im, text="AI GENERATED")
            st.image(im, caption=f"Result {i+1}", use_column_width=True)
            buf = BytesIO()
            im.save(buf, format="PNG")
            byte_data = buf.getvalue()
            st.download_button(label=f"Download image {i+1} (PNG)", data=byte_data, file_name=f"image_{i+1}.png", mime="image/png")
            filenames.append(f"image_{i+1}.png")

        params = {
            "num_images": num_images,
            "guidance_scale": guidance,
            "steps": steps,
            "width": width,
            "height": height,
            "seed": seed_val
        }
        saved_folder = save_images_with_metadata(images, prompt, negative_prompt, params)
        status.success(f"Saved images & metadata to {saved_folder}")



