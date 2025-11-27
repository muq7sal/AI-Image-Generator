# 🖼️ AI-Powered Text-to-Image Generator

# AI Text→Image Generator (Stable Diffusion v1.5)

## What this is
A local text-to-image generator using Stable Diffusion (open-source) with a simple Streamlit UI.
Features:
- Multi-image generation
- Negative prompts
- GPU and CPU fallback
- Watermarking and simple prompt filtering
- Saves images with metadata

## Files
- `app.py` — Streamlit app
- `engine.py` — model loader & generator
- `utils.py` — save / watermark / filter utilities
- `requirements.txt`
- `sample_outputs/` — generated images (created at runtime)

## Setup (Linux/macOS/WSL)
1. Create venv and activate:
   ```bash
   python3 -m venv venv
   source venv/bin/activate

