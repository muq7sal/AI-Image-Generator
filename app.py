import streamlit as st
import torch
import os
from datetime import datetime
from diffusers import StableDiffusionPipeline
from utils import save_image_with_metadata, apply_watermark

st.set_page_config(page_title="AI Image Generator", layout="wide")
st.title("🖼️ AI-Powered Text-to-Image Generator")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device=="cuda" else torch.float32,
        safety_checker=None
    )
    pipe = pipe.to(device)
    return pipe, device

pipe, device = load_model()

# ---------------- User Inputs ----------------
prompt = st.text_input("Enter Prompt", "a futuristic city at sunset, highly detailed, 4K")
negative_prompt = st.text_input("Negative Prompt (optional)", "lowres, blurry, deformed")
num_images = st.slider("Number of Images", 1, 2, 1)
steps = st.slider("Inference Steps", 10, 50, 25)
guidance = st.slider("Guidance Scale", 1.0, 15.0, 7.5)

style = st.selectbox("Select Style", ["None", "Photorealistic", "Cartoon", "Artistic", "Van Gogh"])
style_map = {
    "Photorealistic": "professional photography, ultra realistic, 8K",
    "Cartoon": "Pixar style, clean lines, vibrant colors",
    "Artistic": "digital art, beautifully rendered, detailed lighting",
    "Van Gogh": "oil painting in Van Gogh style, textured brush strokes",
    "None": ""
}

# ---------------- Generate Images ----------------
if st.button("Generate Images"):
    if prompt.strip() == "":
        st.error("Please enter a valid prompt.")
    else:
        style_text = style_map.get(style, "")
        final_prompt = f"{prompt}, {style_text}"
        with st.spinner("Generating... Please wait..."):
            result = pipe(
                prompt=final_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                num_images_per_prompt=num_images
            )
            images = result.images
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("outputs", exist_ok=True)

            st.subheader("Generated Images")
            cols = st.columns(len(images))

            for idx, img in enumerate(images):
                img = apply_watermark(img, "AI Generated")
                filename = f"outputs/output_{timestamp}_{idx}.png"
                metadata = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "style": style,
                    "inference_steps": steps,
                    "guidance_scale": guidance,
                    "timestamp": timestamp
                }
                save_image_with_metadata(img, filename, metadata)
                with cols[idx % len(cols)]:
                    st.image(img, caption=filename, use_column_width=True)
                    st.download_button("Download", data=open(filename, "rb"), file_name=filename)
st.info("Images are automatically saved in the 'outputs' folder.")

