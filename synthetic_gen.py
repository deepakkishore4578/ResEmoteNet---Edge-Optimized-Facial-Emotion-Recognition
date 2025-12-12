import torch
from diffusers import StableDiffusionPipeline
import os

def generate_data():
    # Load Stable Diffusion (Standard v1.5 is good for general realism)
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    # Emotions to augment (Classes with low data in FER-2013)
    target_emotions = ["disgust", "fear"] 
    images_per_class = 500

    for emotion in target_emotions:
        print(f"Generating {images_per_class} images for: {emotion}...")
        save_dir = f"data/train/{emotion}"
        os.makedirs(save_dir, exist_ok=True)
        
        prompt = f"A photo of a human face looking {emotion}, close up, white background, high quality, real photo"
        
        for i in range(images_per_class):
            # Generate
            image = pipe(prompt).images[0]
            # Resize to 48x48 to match FER-2013 (optional here, but saves disk space)
            image = image.resize((48, 48)) 
            # Save
            image.save(f"{save_dir}/syn_{i}.jpg")

if __name__ == "__main__":
    generate_data()