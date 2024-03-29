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

negative_prompt = "The artwork avoids the pitfalls of bad art, such as ugly and deformed eyes and faces, poorly drawn, blurry, and disfigured bodies with extra limbs and close-ups that look weird. It also avoids other common issues such as watermarking, text errors, missing fingers or digits, cropping, poor quality, and JPEG artifacts. The artwork is free of signature or watermark and avoids framing issues. The hands are not deformed, the eyes are not disfigured, and there are no extra bodies or limbs. The artwork is not blurry, out of focus, or poorly drawn, and the proportions are not bad or deformed. There are no mutations, missing limbs, or floating or disconnected limbs. The hands and neck are not malformed, and there are no extra heads or out-of-frame elements. The artwork is not low-res or disgusting and is a well-drawn, highly detailed, and beautiful rendering."

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

# # single generation
# prompt = "A man is sitting in front of a desk with glasses on and rubbing his eyes and the view point is back."
# folder_name = "./examples"
# # run both experts
# image = base(
#     prompt=prompt,
#     negative_prompt = negative_prompt,
#     num_inference_steps=n_steps,
#     denoising_end=high_noise_frac,
#     output_type="latent",
# ).images
# image = refiner(
#     prompt=prompt,
#     negative_prompt = negative_prompt,
#     num_inference_steps=n_steps,
#     denoising_start=high_noise_frac,
#     image=image,
# ).images[0]

# image.save(os.path.join(folder_name,  prompt + "3.png"))

# using for loop
lines_list = []
with open("prompts.txt", "r") as file:
    for line in file:
        lines_list.append(line.strip())

print(lines_list)

folder_name = "./examples/precise_negative_prompt_hand_generation_text_with_various_prompt"
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