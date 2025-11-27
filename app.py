# app.py
import streamlit as st
from engine import TextToImageEngine
from utils import save_images_with_metadata, watermark_image, is_prompt_allowed
from io import BytesIO

st.set_page_config(page_title="AI Text → Image Generator", layout="centered")
st.title("AI Text → Image Generator (Stable Diffusion v1.5)")

# Use HF token from Streamlit secrets
hf_token = st.secrets["HF_API_TOKEN"]

# Sidebar
st.sidebar.header("Settings")
device_choice = st.sidebar.selectbox("Device", ["auto", "cuda", "cpu"], index=0)
num_images = st.sidebar.slider("Number of images", 1, 4, 1)
steps = st.sidebar.slider("Inference steps", 10, 50, 25)
guidance = st.sidebar.slider("Guidance scale", 1.0, 20.0, 7.5)
width = st.sidebar.selectbox("Width", [256, 512], index=1)
height = st.sidebar.selectbox("Height", [256, 512], index=1)
watermark_toggle = st.sidebar.checkbox("Apply watermark", value=True)

# Prompt
prompt = st.text_area("Enter prompt", height=120, placeholder="e.g., A futuristic city at sunset, ultra-detailed, 4k")
negative_prompt = st.text_input("Negative prompt (optional)")

# Generate
if st.button("Generate"):
    if not prompt.strip():
        st.error("Please enter a prompt!")
    else:
        allowed, reason = is_prompt_allowed(prompt)
        if not allowed:
            st.error(reason)
        else:
            device_param = None if device_choice=="auto" else device_choice
            if "engine" not in st.session_state:
                with st.spinner("Loading model (may take ~1 min)..."):
                    st.session_state["engine"] = TextToImageEngine(hf_token=hf_token, device=device_param)
            engine = st.session_state["engine"]

            status = st.empty()
            status.info("Generating images...")

            images = engine.generate(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt.strip() else None,
                num_images=num_images,
                guidance_scale=guidance,
                num_inference_steps=steps,
                height=height,
                width=width
            )

            for idx, img in enumerate(images):
                if watermark_toggle:
                    img = watermark_image(img)
                st.image(img, caption=f"Result {idx+1}", use_column_width=True)
                buf = BytesIO()
                img.save(buf, format="PNG")
                st.download_button(f"Download image {idx+1}", buf.getvalue(), file_name=f"image_{idx+1}.png")

            folder = save_images_with_metadata(images, prompt, negative_prompt, {"num_images":num_images,"guidance":guidance,"steps":steps,"width":width,"height":height})
            status.success(f"Saved images & metadata to {folder}")
