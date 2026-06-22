import multiprocessing


if __name__ == '__main__':
    multiprocessing.freeze_support()

    import torch
    from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Activate the GPU-enabled ddpm environment or check the NVIDIA driver.")

    # Mammography: single-channel, 16-bit images
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = 1,
        flash_attn = False
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 1024,
        timesteps = 1000,
        sampling_timesteps = 250
    )

    trainer = Trainer(
        diffusion,
        r"data\\mass\\trainB",
        train_batch_size = 1,
        train_lr = 8e-5,
        train_num_steps = 200000,
        gradient_accumulate_every = 16,
        ema_decay = 0.995,
        amp = True,
        augment_horizontal_flip = False,
        convert_image_to = "I;16",
        calculate_fid = False
    )

    trainer.train()
