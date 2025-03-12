import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np

from time import time
import logging
import os

from diffusers.models import AutoencoderKL
from efficientvit.ae_model_zoo import DCAE_HF
from models import FLAV_models

from accelerate.logging import get_logger
from diffusion.rectified_flow import RectifiedFlow

from diffusers.training_utils import EMAModel
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

import einops
from common_parser import CommonParser
from utils import *
from converter import  denormalize, denormalize_spectrogram, Generator

AUDIO_T_PER_FRAME = 1600 // 160 


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = get_logger(__name__)
    return logger

@torch.no_grad()
def encode_video(video, vae):
    b, t, c, h, w = video.shape
    video = einops.rearrange(video, "b t c h w-> (b t) c h w")

    video_enc = []
    for i in range(0, video.shape[0], b):
        if isinstance(vae, AutoencoderKL):
            video_enc.append(vae.encode(video[i:i+b]).latent_dist.sample().mul_(0.18215))
        else:
            video_enc.append(vae.encode(video[i:i+b]) * vae.cfg.scaling_factor)

    video = torch.cat(video_enc, dim=0)
    video = einops.rearrange(video, "(b t) c h w -> b t c h w", t=t)
    return video

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new FLAV model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    config_no_save = args.config_no_save
    config_name = args.config_name
    iterations = args.iterations

    os.makedirs(args.experiment_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    model_string_name = args.model.replace("/", "-")  
    experiment_dir = f"{args.experiment_dir}/{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    os.makedirs(os.path.join(experiment_dir, args.results_dir), exist_ok=True)

    if not config_no_save:
        dict_to_json(os.path.join(experiment_dir, config_name), args)
    
    #Read if there is a checkpoint in the given directory
    try:
        checkpoint_files = os.listdir(checkpoint_dir)
        checkpoint_files = [int(x.split("_")[-1]) for x in checkpoint_files]
        checkpoint_files.sort()
        checkpoint_file = checkpoint_files[-1]
    except:
        checkpoint_file = 0

    accelerator_project_config = ProjectConfiguration(project_dir=experiment_dir, automatic_checkpoint_naming=True, iteration = checkpoint_file+1, total_limit=args.ckpt_limit)
    accelerator = Accelerator(project_config=accelerator_project_config)
    
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    if args.load_config:
        #Do not change experiment dir and results dir
        args = json_to_dict(args.config_path, args)
    log_args(args, logger)

    device = accelerator.device 

    set_seed(args.seed)  # Set global seed for reproducibility

    if args.use_sd_vae:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}")
    else:
        vae = DCAE_HF.from_pretrained(f"mit-han-lab/dc-ae-f32c32-sana-1.0")

    assert args.image_size % (8 if args.use_sd_vae else 32) == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // (8 if args.use_sd_vae else 32)

    video_latent_size = (1, args.predict_frames, 4 if args.use_sd_vae else 32, latent_size, latent_size)
    audio_latent_size = (1, args.predict_frames, 1, 256, AUDIO_T_PER_FRAME)
    
    model = FLAV_models[args.model](
        latent_size=latent_size,
        in_channels = (4 if args.use_sd_vae else 32),
        num_classes=args.num_classes,
        predict_frames = args.predict_frames,
        grad_ckpt = args.grad_ckpt,
        causal_attn = args.causal_attn,
    )

    ema = EMAModel(model.parameters(), power=3./4.) 

    rectified_flow = RectifiedFlow(num_timesteps=args.num_timesteps, 
                                   warmup_timesteps=args.predict_frames, 
                                   window_size=args.predict_frames, 
                                   sampling=args.sampling)

    logger.info(f"FLAV Parameters: {sum(p.numel() for p in model.parameters()):,}")

    vocoder = Generator.from_pretrained(args.vocoder_ckpt)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.optimizer_wd)

    loader = get_dataloader(args, logger, args.predict_frames, train=True, latents=args.load_latents)

    model, opt, loader = accelerator.prepare(model, opt, loader)
    
    vocoder.to(device)
    vae.to(device)
    ema.to(device)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0

    def save_model_hook(models, weights, output_dir):
        accelerator.save(ema.state_dict(), os.path.join(output_dir,"ema.pth"))

    def load_model_hook(models, input_dir):
        ema.load_state_dict(torch.load(os.path.join(input_dir,"ema.pth"),map_location="cpu"))
        ema.to(device)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    try:
        train_steps = checkpoint_file*args.ckpt_every
        log_steps = train_steps % args.log_every
        accelerator.load_state()
        logger.info(f"Checkpoint found, starting from {checkpoint_file*args.ckpt_every} steps.")
    except Exception as e:
        # accelerator.save_state()
        logger.info("No checkpoint found, starting from scratch.")
        logger.info(e)
    
    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    running_loss = 0
    start_time = time()

    logger.info(f"Training for {iterations} steps max...")

    while train_steps <= iterations:
        for v, a, y in loader: # video shape is B T C H W, audio shape is B 1 N_MEL T
            with torch.no_grad():
                # Map input images to latent space
                if not args.load_latents:
                    v = encode_video(v, vae) # B T C H W
                a = einops.rearrange(a, "B C N (T F) -> B T C N F", T=args.predict_frames)

                a = args.audio_scale * a
            
            model_kwargs = dict(y=y)
            loss_dict = rectified_flow.training_loss(model, v, a, model_kwargs)
    
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            running_loss += loss.item()
            
            # Log loss values:
            # Wait for all processes to update their weights values:
            if accelerator.sync_gradients:
                ema.step(model.parameters())

                log_steps += 1
                train_steps += 1
                
                if train_steps % args.log_every == 0:
                    # Measure training speed:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    # Reset monitoring variables:
                    running_loss = 0
                    log_steps = 0
                    start_time = time()
    
                # Save VAAR checkpoint:
                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    accelerator.save_state()
    
                if train_steps % args.sample_every == 0 and train_steps > 0 and accelerator.is_main_process:
                    torch.cuda.empty_cache()
                    model.eval()
    
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
    
                    video, audio = generate_sample(vae = vae, 
                                                   rectified_flow = rectified_flow, 
                                                   forward_fn = model.forward, 
                                                   video_length = args.video_length, 
                                                   video_latent_size = video_latent_size,
                                                   audio_latent_size = audio_latent_size,
                                                   y = torch.randint(args.num_classes, (1,)).to(device) if args.num_classes > 0 else torch.tensor([0]*8),
                                                   cfg_scale = None,
                                                   device = device)
                    
                    ema.restore(model.parameters())
                    model.train()
    
                    # Save and display images:
                    os.makedirs(os.path.join(experiment_dir,"results/",f"{train_steps}/"), exist_ok=True)
                    wavs = get_wavs(audio, vocoder, device, args.audio_scale)
                    for i,(vid, wav) in enumerate(zip(video, wavs)):
                        save_multimodal(vid, wav, os.path.join(experiment_dir,"results/",f"{train_steps}/"), f"sample_{i}")
    
                    torch.cuda.empty_cache()
                
    logger.info(f"Maximum iterations reached...")


    accelerator.wait_for_everyone()   
    if accelerator.is_main_process:
        checkpoint_path = f"{checkpoint_dir}/final.pt"

        accelerator.save({
            "model": accelerator.unwrap_model(model).state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.optimizer.state_dict(),
            "args": args
            }, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        logger.info("Done!")


def get_wavs(norm_spec, vocoder, device, audio_scale = 1):
    norm_spec = norm_spec.squeeze(1)
    norm_spec = norm_spec / audio_scale
    post_norm_spec = denormalize(norm_spec).to(device)
    raw_chunk_spec = denormalize_spectrogram(post_norm_spec)
    wavs = vocoder.inference(raw_chunk_spec)
    return wavs


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = CommonParser().get_parser()
    parser.add_argument("--model-ckpt", type=str)
    parser.add_argument("--max-grad-norm", type=float, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=400000)
    parser.add_argument("--ckpt-limit", type=int, default=2)
    parser.add_argument("--sample-every", type=int, default=5000)
    parser.add_argument("--grad-ckpt", action="store_true")

    parser.add_argument("--sampling", type=str, choices=["uniform", "logit"], default="uniform")

    args = parser.parse_args()
    main(args)
