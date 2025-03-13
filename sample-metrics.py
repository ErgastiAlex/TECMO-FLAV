import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import os

from diffusers.models import AutoencoderKL
from efficientvit.ae_model_zoo import DCAE_HF
from models import FLAV

from diffusion.rectified_flow import RectifiedFlow
from accelerate.utils import set_seed
from converter import Generator
from huggingface_hub import hf_hub_download

from utils import *

AUDIO_T_PER_FRAME = 1600 // 160 

#################################################################################
#                                  Sampling Loop                                #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Sampling currently requires at least one GPU."
    os.makedirs(args.results_dir, exist_ok=True)
    device = "cuda"
    set_seed(args.seed)  # Set global seed for reproducibility
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema")
    latent_size = 256 // 8


    model = FLAV.from_pretrained(args.model_ckpt)

    hf_hub_download(repo_id=args.model_ckpt, filename="vocoder/config.json")
    vocoder_path = hf_hub_download(repo_id=args.model_ckpt, filename="vocoder/vocoder.pt")

    vocoder_path = vocoder_path.replace("vocoder.pt", "")
    vocoder = Generator.from_pretrained(vocoder_path)

    rectified_flow = RectifiedFlow(num_timesteps=args.num_timesteps, 
                                   warmup_timesteps=model.predict_frames,
                                   window_size=model.predict_frames)
    

    model.to(device)    
    vocoder.to(device)
    vae.to(device)

    os.makedirs(os.path.join(args.results_dir, "samples/"), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "samples/audio"), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "samples/video"), exist_ok=True)

    video_latent_size = (args.batch_size, model.predict_frames, 4, latent_size, latent_size)
    audio_latent_size = (args.batch_size, model.predict_frames, 1, 256, AUDIO_T_PER_FRAME)
    
    for i in range(args.num_videos//args.batch_size):
        video, audio = generate_sample(vae = vae, 
            rectified_flow = rectified_flow, 
            forward_fn = model.forward_with_cfg if args.num_classes > 0 else model.forward, 
            video_length = args.video_length, 
            video_latent_size = video_latent_size,
            audio_latent_size = audio_latent_size,
            y = torch.Tensor(args.classes, device="cuda") if args.num_classes > 0 else None,
            cfg_scale = args.cfg_scale if args.num_classes > 0 else None,
            device = device)

        wavs = get_wavs(audio, vocoder, args.audio_scale, device)
        for j,(vid, wav) in enumerate(zip(video, wavs)):
            save_multimodal(vid, wav, os.path.join(args.results_dir, "samples/"), f"sample_{i*(args.batch_size)+j}")



import argparse
if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", type=str)

    parser.add_argument("--audio-scale", type=float, default=3.5009668382765917)

    parser.add_argument("--num-timesteps", type=int, default=2)

    parser.add_argument("--cfg-scale", type=float, default=1)
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--classes", type=int, nargs="+", default=[0])

    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--video-length", type=int, default=1)
    parser.add_argument("--num-videos", type=int, default=2048)

    
    args = parser.parse_args()
    main(args)
