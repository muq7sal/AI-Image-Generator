# app.py
import streamlit as st
from engine import TextToImageEngine
from utils import save_images_with_metadata, watermark_image, is_prompt_allowed
from io import BytesIO

st.set_page_config(page_title="Text→Image Generator", layout="centered")
st.title("AI Text→Image Generator (Stable Diffusion)")

# Load Hugging Face token from secrets
HF_TOKEN = st.secrets["HF_API_TOKEN"]

# Sidebar options
st.sidebar.header("Settings")
device_choice = st.sidebar.selectbox("Device", ["auto", "cuda", "cpu"], index=0)
model_id = st.sidebar.text_input("Model ID", "runwayml/stable-diffusion-v1-5")

default_steps = st.sidebar.slider("Inference steps", 10, 100, 30)
default_guidance = st.sidebar.slider("Guidance scale", 1.0, 20.0, 7.5)
default_num = st.sidebar.slider("Images per prompt", 1, 4, 1)

# Input form
with st.form("generate_form"):
    prompt = st.text_area("Prompt", height=140, placeholder="A futuristic city at sunset, ultra-detailed, 4k")
    negative_prompt = st.text_input("Negative prompt (optional)")
    num_images = st.slider("Number of images", 1, 4, default_num)
    steps = st.number_input("Inference steps", 1, 200, default_steps)
    guidance = st.number_input("Guidance scale", 1.0, 30.0, float(default_guidance))
    seed = st.number_input("Seed (0=random)", 0, 1000000, 0)
    watermark_toggle = st.checkbox("Apply watermark", True)
    submitted = st.form_submit_button("Generate")

if submitted:
    allowed, reason = is_prompt_allowed(prompt)
    if not allowed:
        st.error(f"Prompt blocked: {reason}")
    else:
        device_param = None if device_choice=="auto" else device_choice
        if "engine" not in st.session_state or st.session_state.get("engine_model") != model_id:
            try:
                with st.spinner("Loading model..."):
                    st.session_state["engine"] = TextToImageEngine(model_id, device=device_param, hf_token=HF_TOKEN)
                    st.session_state["engine_model"] = model_id
            except Exception as e:
                st.error(f"Error loading model: {e}")
        engine = st.session_state.get("engine")
        if engine:
            try:
                status = st.empty()
                status.info("Generating image(s)...")
                seed_val = None if seed==0 else int(seed)
                images = engine.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt.strip() else None,
                    num_images=num_images,
                    guidance_scale=guidance,
                    num_inference_steps=steps
                )
                for i, img in enumerate(images):
                    if watermark_toggle:
                        img = watermark_image(img)
                    st.image(img, caption=f"Result {i+1}", use_column_width=True)
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    st.download_button(f"Download image {i+1}", buf.getvalue(), f"image_{i+1}.png", "image/png")
                save_images_with_metadata(images, prompt, negative_prompt, {"steps":steps,"guidance":guidance,"num_images":num_images})
                status.success("Images generated successfully!")
            except Exception as e:
                st.error(f"Error generating image: {e}")
