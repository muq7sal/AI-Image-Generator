# engine.py
import torch
from diffusers import StableDiffusionPipeline
from typing import List, Optional

DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"

class TextToImageEngine:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: Optional[str] = None):
        # Auto detect device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # dtype
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # load model
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch_dtype
        )

        # Move to device
        self.pipe = self.pipe.to(self.device)

        # Optimize for memory
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
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
        seed: Optional[int] = None,
        progress_callback=None
    ) -> List:
        # Seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # If progress_callback provided, pass to pipeline
        if progress_callback:
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                generator=generator,
                callback=progress_callback,
                callback_steps=1
            )
        else:
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
