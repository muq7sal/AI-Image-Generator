# app.py
import streamlit as st
from engine import TextToImageEngine
from utils import save_images_with_metadata, watermark_image, is_prompt_allowed
from io import BytesIO

st.set_page_config(page_title="Text→Image Generator", layout="centered")
st.title("AI Text → Image Generator (Stable Diffusion)")

# Sidebar
st.sidebar.header("Model & Hardware")
device_choice = st.sidebar.selectbox("Device", ["auto", "cuda", "cpu"], index=0)
model_id = st.sidebar.text_input("Model ID", value="runwayml/stable-diffusion-v1-5")

st.sidebar.header("Generation defaults")
default_steps = st.sidebar.slider("Inference steps", 10, 100, 30)
default_guidance = st.sidebar.slider("Guidance scale", 1.0, 20.0, 7.5)
default_num = st.sidebar.slider("Default images per prompt", 1, 4, 1)

# Fast preview option
fast_preview = st.sidebar.checkbox("Fast Preview (low-res / fewer steps)")

# Main form
with st.form("generate_form"):
    prompt = st.text_area("Prompt", height=120, placeholder="A futuristic city at sunset, ultra-detailed, 4k")
    negative_prompt = st.text_input("Negative prompt (optional)")
    num_images = st.slider("Number of images", 1, 4, default_num)
    steps = st.number_input("Inference steps", 1, 200, default_steps)
    guidance = st.number_input("Guidance scale", 1.0, 30.0, float(default_guidance))
    width = st.selectbox("Width", [256, 512, 640, 768], index=1)
    height = st.selectbox("Height", [256, 512, 640, 768], index=1)
    seed = st.number_input("Seed (0=random)", value=0, step=1)
    watermark_toggle = st.checkbox("Apply 'AI GENERATED' watermark", value=True)
    submitted = st.form_submit_button("Generate")

if submitted:
    allowed, reason = is_prompt_allowed(prompt)
    if not allowed:
        st.error(f"Prompt blocked: {reason}")
    else:
        # Apply fast preview overrides
        if fast_preview:
            steps = min(steps, 15)
            width = min(width, 256)
            height = min(height, 256)
            num_images = min(num_images, 1)

        # Determine device
        device_param = None if device_choice=="auto" else device_choice

        # Lazy-load engine
        if "engine" not in st.session_state or st.session_state.get("engine_model") != model_id:
            with st.spinner("Loading model (may take a minute)..."):
                st.session_state["engine"] = TextToImageEngine(model_id=model_id, device=device_param)
                st.session_state["engine_model"] = model_id

        engine: TextToImageEngine = st.session_state["engine"]

        # Progress display
        progress_text = st.empty()
        def progress_callback(step, timestep, latents):
            progress_text.text(f"Generating... Step {step}/{steps}")

        # Generate images
        imgs = engine.generate(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_images=num_images,
            guidance_scale=guidance,
            num_inference_steps=steps,
            width=width,
            height=height,
            seed=None if seed==0 else int(seed),
            progress_callback=progress_callback
        )

        # Display + download
        filenames = []
        for i, pil_img in enumerate(imgs):
            if watermark_toggle:
                pil_img = watermark_image(pil_img, text="AI GENERATED")
            st.image(pil_img, caption=f"Result {i+1}", use_column_width=True)
            buf = BytesIO()
            pil_img.save(buf, format="PNG")
            st.download_button(
                f"Download image {i+1}", buf.getvalue(), f"image_{i+1}.png", "image/png"
            )
            filenames.append(f"image_{i+1}.png")

        meta_params = {
            "num_images": num_images, "guidance_scale": guidance,
            "steps": steps, "width": width, "height": height,
            "seed": None if seed==0 else int(seed)
        }
        saved_folder = save_images_with_metadata(imgs, prompt, negative_prompt, meta_params)
        progress_text.success(f"Saved images & metadata to {saved_folder}")
