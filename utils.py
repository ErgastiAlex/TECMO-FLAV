# from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
# # from moviepy.audio.AudioClip import AudioArrayClip
# from moviepy.audio.io.AudioFileClip import AudioFileClip
from torch.utils.data import DataLoader
from dataset import AudioVideoDataset, LatentDataset
import torch as th
import numpy as np

import einops
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from diffusers.models import AutoencoderKL

from converter import denormalize, denormalize_spectrogram

import soundfile as sf
import os
import json
import torch
from tqdm import tqdm
#################################################################################
#                                  Video Utils                                  #
#################################################################################


def preprocess_video(video):
    # video = 255*(video+1)/2.0 # [-1,1] -> [0,1] -> [0,255]
    # video = th.clamp(video, 0, 255).to(dtype=th.uint8, device="cuda")
    video = out2img(video)
    video = einops.rearrange(video, 't c h w -> t h w c').cpu().numpy()
    return video

def preprocess_video_batch(videos):
    B = videos.shape[0]
    videos_prep = np.empty(B, dtype=np.ndarray)
    for b in range(B):
        videos_prep[b] = preprocess_video(videos[b])
    videos_prep = np.stack(videos_prep, axis=0)
    return videos_prep

def save_latents(video, audio, y, output_path, name_prefix, ext=".pt"):
    os.makedirs(output_path, exist_ok=True)
    th.save(
        {
        "video":video,
        "audio":audio,
        "y":y
        }, os.path.join(output_path, name_prefix + ext))

def save_multimodal(video, audio, output_path, name_prefix, video_fps=10, audio_fps=16000, audio_dir=None):
    if not audio_dir:
        audio_dir = output_path

    #prepare folders
    audio_dir = os.path.join(audio_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, name_prefix + "_audio.wav")

    video_dir = os.path.join(output_path, "video")
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, name_prefix + "_video.mp4")

    #save audio
    sf.write(audio_path, audio, samplerate=audio_fps)

    #save video
    video = preprocess_video(video)

    imgs = [img for img in video]    
    video_clip = ImageSequenceClip(imgs, fps=video_fps)
    audio_clip = AudioFileClip(audio_path)
    video_clip = video_clip.with_audio(audio_clip)   
    video_clip.write_videofile(video_path, video_fps, audio=True, audio_fps=audio_fps)

def get_dataloader(args, logger, sequence_length, train, latents=False):
    if latents:
        train_set = LatentDataset(args.data_path, train=train)
    else:
        train_set = AudioVideoDataset(
            args.data_path, 
            train=train, 
            sample_every_n_frames=1, 
            resolution=args.image_size, 
            sequence_length = sequence_length, 
            audio_channels = 1, 
            sample_rate=16000, 
            min_length=1,
            ignore_cache=args.ignore_cache,
            labeled=args.num_classes > 0,
            target_video_fps=args.target_video_fps,
        )
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )  
    if logger is not None:
        logger.info(f'{"Train" if train else "Test"} Dataset contains {len(train_set)}, images ({args.data_path})')
    else:
        print(f'{"Train" if train else "Test"} Dataset contains {len(train_set)}, images ({args.data_path})')
    return loader

@torch.no_grad()
def encode_video(video, vae, use_sd_vae = False):
    b, t, c, h, w = video.shape
    video = einops.rearrange(video, "b t c h w-> (b t) c h w")
    if use_sd_vae:
        video = vae.encode(video).latent_dist.sample().mul_(0.18215)
    else:
        video = vae.encode(video)*vae.cfg.scaling_factor
    video = einops.rearrange(video, "(b t) c h w -> b t c h w", t=t)
    return video

@torch.no_grad()
def decode_video(video, vae):
    b = video.shape[0]
    video_decoded = []
    video = einops.rearrange(video, "b t c h w -> (b t) c h w")
    
    #use minibatch to avoid memory error
    for i in range(0, video.shape[0], b):
        if isinstance(vae, AutoencoderKL):
            video_decoded.append(vae.decode(video[i:i+b] / 0.18215).sample.detach().cpu())
        else:
            video_decoded.append(vae.decode(video[i:i+b] / vae.cfg.scaling_factor).detach().cpu())
    
    video = torch.cat(video_decoded, dim=0)
    video = einops.rearrange(video, "(b t) c h w ->b t c h w",b=b)
    return video


def generate_sample(vae, 
                    rectified_flow, 
                    forward_fn, 
                    video_length, 
                    video_latent_size,
                    audio_latent_size,
                    y,
                    cfg_scale,
                    device):
    

    with torch.no_grad():
        v_z = torch.randn(video_latent_size, device=device)*rectified_flow.noise_scale
        a_z = torch.randn(audio_latent_size, device=device)*rectified_flow.noise_scale

        model_kwargs = dict(y=y, cfg_scale=cfg_scale) if cfg_scale else dict(y=y)

        sample_fn = rectified_flow.sample(
                    forward_fn, v_z, a_z, model_kwargs=model_kwargs, progress=True)()
        
        video = []
        audio = []
        for _ in tqdm(range(video_length), desc="Generating frames"):
            video_samples, audio_samples = next(sample_fn)

            video.append(video_samples)
            audio.append(audio_samples)
        
        video = torch.stack(video, dim=1)
        audio = torch.stack(audio, dim=1)
        
        video = decode_video(video, vae)
        audio = einops.rearrange(audio, "B T C N F -> B C N (T F)")

        return video, audio
    
def generate_sample_a2v(vae, 
                    rectified_flow, 
                    forward_fn, 
                    video_length, 
                    video_latent_size,
                    audio,
                    y,
                    device,
                    cfg_scale=1,
                    scale=1):
    

    v_z = torch.randn(video_latent_size, device=device)*rectified_flow.noise_scale

    model_kwargs = dict(y=y, cfg_scale=cfg_scale) if cfg_scale else dict(y=y)
    
    sample_fn = rectified_flow.sample_a2v(
                forward_fn, v_z, audio, model_kwargs=model_kwargs, scale=scale, progress=True)()
    
    video = []
    for i in tqdm(range(video_length), desc="Generating frames"):
        video_samples = next(sample_fn)

        video.append(video_samples)
    
    video = torch.stack(video, dim=1)
    
    video = decode_video(video, vae)
    audio = einops.rearrange(audio, "B T C N F -> B C N (T F)")

    return video, audio
    
def generate_sample_v2a(vae, 
                    rectified_flow, 
                    forward_fn, 
                    video_length, 
                    video,
                    audio_latent_size,
                    y,
                    device,
                    cfg_scale=1,
                    scale=1):
    

    a_z = torch.randn(audio_latent_size, device=device)*rectified_flow.noise_scale
    
    model_kwargs = dict(y=y, cfg_scale=cfg_scale) if cfg_scale else dict(y=y)
    
    sample_fn = rectified_flow.sample_v2a(
                forward_fn, video, a_z, model_kwargs=model_kwargs, scale=scale, progress=True)()
    
    audio = []
    for i in tqdm(range(video_length), desc="Generating frames"):
        audio_samples = next(sample_fn)

        audio.append(audio_samples)
    
    audio = torch.stack(audio, dim=1)
    
    video = decode_video(video, vae)
    audio = einops.rearrange(audio, "B T C N F -> B C N (T F)")

    return video, audio

def dict_to_json(path, args):
    with open(path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
def json_to_dict(path, args):
    with open(path, 'r') as f:
        args.__dict__ = json.load(f)
    return args

def log_args(args, logger):
    text = ""
    for k, v in vars(args).items():
        text += f'{k}={v}\n'
    logger.info(f"##### ARGS #####\n{text}")

def out2img(samples):
    return th.clamp(127.5 * samples + 128.0, 0, 255).to(
        dtype=th.uint8
    ).cuda()

def get_gpu_usage():
    device = th.device('cuda:0')
    free, total = th.cuda.mem_get_info(device)
    mem_used_MB = (total - free) / 1024 ** 2
    return mem_used_MB

def get_wavs(norm_spec, vocoder, audio_scale, device):
    norm_spec = norm_spec.squeeze(1)
    norm_spec = norm_spec / audio_scale
    post_norm_spec = denormalize(norm_spec).to(device)
    raw_chunk_spec = denormalize_spectrogram(post_norm_spec)
    wavs = vocoder.inference(raw_chunk_spec)
    return wavs