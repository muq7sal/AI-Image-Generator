# 🖼️ AI-Powered Text-to-Image Generator (Stable Diffusion via Hugging Face Router API)

## Overview
This project is a **text-to-image generator** using **Stable Diffusion v1.5** and the **Hugging Face Router API**.  
It allows users to enter prompts, optionally specify negative prompts, generate multiple images, apply watermarks, and download results. Images are also saved with metadata (prompt, negative prompt, parameters, timestamp).

### Features
- Generates high-quality images from text prompts
- Supports negative prompts for better output
- GPU/CPU fallback via Hugging Face Router API
- Watermarking option
- Saves images with metadata in organized folders
- Simple Streamlit UI

## Files
- `app.py` — Streamlit app (UI + HF API integration)
- `utils.py` — Utilities for saving images, watermarking, prompt filtering
- `requirements.txt` — Dependencies
- `sample_outputs/` — Generated images (created at runtime)

## Setup & Deployment

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Hugging Face API Token
- On **Streamlit Cloud**:
  1. Go to **Manage App → Advanced Settings → Secrets**
  2. Add:
     ```
     HF_API_TOKEN = "your_huggingface_token_here"
     ```
- Locally:
  ```bash
  export STREAMLIT_SECRETS='{"HF_API_TOKEN": "your_huggingface_token_here"}'
  ```

### 4. Run Locally
```bash
streamlit run app.py
```

### 5. Usage
1. Enter your text prompt
2. Optionally enter negative prompts to filter unwanted elements
3. Choose the number of images
4. Generate images
5. Download results or view saved images in `sample_outputs/`

### 6. Example Prompts
- **Prompt:** `"A futuristic city at sunset, ultra-detailed, cinematic lighting, 4K"`
- **Negative Prompt:** `"blurry, low quality, watermark, text, distorted, artifacts"`

### Notes
- Using **Hugging Face Router API** ensures your app works reliably without deprecated endpoints.
- Watermarks indicate AI-generated images.
- Use ethically responsible prompts only.
