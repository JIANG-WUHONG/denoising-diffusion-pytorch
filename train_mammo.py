import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

# Mammography: single-channel, 16-bit images
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 1,
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 512,
    timesteps = 1000,
    sampling_timesteps = 250
)

trainer = Trainer(
    diffusion,
    r"D:\\path\\to\\mammo_images",
    train_batch_size = 8,
    train_lr = 8e-5,
    train_num_steps = 200000,
    gradient_accumulate_every = 2,
    ema_decay = 0.995,
    amp = True,
    augment_horizontal_flip = False,
    convert_image_to = "I;16",
    calculate_fid = False
)

trainer.train()
