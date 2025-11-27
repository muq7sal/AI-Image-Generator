# engine.py
import torch
from diffusers import StableDiffusionPipeline
from typing import List, Optional
from PIL import Image

DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"

class TextToImageEngine:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: Optional[str] = None, hf_token: Optional[str] = None):
        """
        Initialize Stable Diffusion pipeline
        Args:
            model_id: Hugging Face model ID
            device: "cuda", "cpu", or None (auto-detect)
            hf_token: Hugging Face API token
        """
        # Detect device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # dtype: float16 for GPU, float32 for CPU
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Load Stable Diffusion pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,       # optional: can add later
            use_auth_token=hf_token    # <--- important: token for Hugging Face
        )

        # Move pipeline to device
        self.pipe = self.pipe.to(self.device)

        # Optimize memory on GPU
        if self.device == "cuda":
            try:
                self.pipe.enable_attention_slicing()
            except Exception:
                pass
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_images: int = 1,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Generate images from prompt
        Args:
            prompt: Text prompt
            negative_prompt: Optional negative prompt to filter results
            num_images: Number of images to generate
            guidance_scale: How strongly the model follows prompt
            num_inference_steps: Diffusion steps
            height: Image height
            width: Image width
            seed: Optional random seed for reproducibility
        Returns:
            List of PIL images
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            generator=generator
        )

        return result.images
