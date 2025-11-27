# engine.py
import torch
from diffusers import StableDiffusionPipeline

def load_model(model_name="runwayml/stable-diffusion-v1-5", device=None):
    """
    Load Stable Diffusion model with GPU/CPU fallback (CPU for Streamlit Cloud).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float32 if device == "cpu" else torch.float16
    pipeline = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=dtype)
    pipeline = pipeline.to(device)
    return pipeline

def generate_images(pipeline, prompt, num_images=1, guidance_scale=7.5):
    """
    Generate images from text prompt.
    """
    images = []
    for _ in range(num_images):
        img = pipeline(prompt, guidance_scale=guidance_scale).images[0]
        images.append(img)
    return images
