# engine.py
import torch
from diffusers import StableDiffusionPipeline
from typing import List, Optional
from PIL import Image

DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"

class TextToImageEngine:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: Optional[str] = None, hf_token: Optional[str] = None):
        # Detect device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Load Stable Diffusion pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,
            use_auth_token=hf_token
        )
        self.pipe = self.pipe.to(self.device)

        # GPU optimizations
        if self.device == "cuda":
            try: self.pipe.enable_attention_slicing()
            except: pass
            try: self.pipe.enable_xformers_memory_efficient_attention()
            except: pass

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
