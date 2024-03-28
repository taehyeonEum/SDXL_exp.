from diffusers import DiffusionPipeline
import torch
import os

# load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to("cuda")
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

prompt = "A man holding a balloon with a heart on it wants to fall in love again because I feel a little bored."

lines_list = []
with open("prompts.txt", "r") as file:
    for line in file:
        lines_list.append(line.strip())

print(lines_list)

folder_name = "./examples/negative_prompt_hand_generation_text_with_various_prompt"
os.mkdir(folder_name)

for idx, prompt in enumerate(lines_list):
    # run both experts
    image = base(
        prompt=prompt,
        negative_prompt = "bad hands",
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        negative_prompt = "bad hands",
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]

    image.save(os.path.join(folder_name,  prompt + ".png"))