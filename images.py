import os
from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path

# Create output directory
IMG_DIR = Path("data/images")
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Hugging Face token from environment
HF_TOKEN = os.getenv("HF_TOKEN", None)

def load_sd(model_id="runwayml/stable-diffusion-v1-5"):
    auth = {"use_auth_token": HF_TOKEN} if HF_TOKEN else {}
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        **auth
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return pipe

def generate_image_for_prompt(pipe, prompt, out_path, height=512, width=512, guidance_scale=7.5, num_inference_steps=25):
    image = pipe(
        prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images[0]
    image.save(out_path)
    return out_path

def get_next_filename():
    """Automatically find the next available demo#.png name."""
    existing = list(IMG_DIR.glob("demo*.png"))
    if not existing:
        return IMG_DIR / "demo1.png"
    nums = [
        int(f.stem.replace("demo", "")) 
        for f in existing if f.stem.replace("demo", "").isdigit()
    ]
    next_num = max(nums, default=0) + 1
    return IMG_DIR / f"demo{next_num}.png"

if __name__ == "__main__":
    # Load model
    pipe = load_sd()

    # Prompt
    prompt = input("Enter image prompt: ")

    # Auto–generate filename
    out_path = get_next_filename()

    # Generate and save
    generate_image_for_prompt(pipe, prompt, out_path)
    print(f"✅ Saved {out_path}")
