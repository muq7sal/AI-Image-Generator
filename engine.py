# engine.py
import torch
from diffusers import StableDiffusionPipeline
from typing import List, Optional

DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"

class TextToImageEngine:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: Optional[str] = None):
        # detect device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # load model
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch_dtype
        ).to(self.device)

        # memory optimizations
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

    def suggest_fast_settings(self):
        """Returns recommended width, height, steps, num_images based on device/VRAM"""
        if self.device == "cpu":
            return {"width": 256, "height": 256, "steps": 15, "num_images": 1}
        else:
            # GPU, check memory
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_gb = meminfo.total / 1024**3
            except Exception:
                total_gb = 8  # fallback

            if total_gb < 6:
                return {"width": 512, "height": 512, "steps": 20, "num_images": 1}
            else:
                return {"width": 768, "height": 768, "steps": 30, "num_images": 2}

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
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

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
