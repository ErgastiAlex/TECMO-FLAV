import argparse
from models import FLAV_models

class CommonParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # Datasets
        self.parser.add_argument("--data-path", type=str, required=True)
        self.parser.add_argument("--load-latents", action="store_true")
        self.parser.add_argument("--num-classes", type=int, default=9)
        self.parser.add_argument("--image-size", type=int, choices=[64, 256, 512, 1024], default=256)
        self.parser.add_argument("--target-video-fps", type=int, default=10)
        self.parser.add_argument("--ignore-cache", action="store_true")
        self.parser.add_argument("--audio-scale", type=float, default=3.5009668382765917)
        
        # Results
        self.parser.add_argument("--video-length", type=int, default=1)
        self.parser.add_argument("--predict-frames", type=int, default=10)
        self.parser.add_argument("--results-dir", type=str, default="results")
        self.parser.add_argument("--experiment-dir", type=str, default="")
        self.parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
        self.parser.add_argument("--ckpt-every", type=int, default=5_000)
        
        # Models
        self.parser.add_argument("--seed", type=int, default=42)
        self.parser.add_argument("--model", type=str, choices=list(FLAV_models.keys()), default="FLAV-XL/2")
        self.parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
        self.parser.add_argument("--use_sd_vae", action="store_true")
        self.parser.add_argument("--vocoder-ckpt", type=str, default="vocoder/")
        self.parser.add_argument("--optimizer-wd", type=float, default=0.02)
        
        # Resources
        self.parser.add_argument("--batch-size", type=int, default=4)
        self.parser.add_argument("--num-workers", type=int, default=32)
        self.parser.add_argument("--log-every", type=int, default=100)
        
        # Config
        self.parser.add_argument("--load-config", action="store_true")
        self.parser.add_argument("--config-no-save", action="store_true")
        self.parser.add_argument("--config-path", type=str, default="")
        self.parser.add_argument("--config-name", type=str, default="config.json")
        
        # Architecture
        self.parser.add_argument("--causal-attn", action="store_true")
        
        #RF
        self.parser.add_argument("--num_timesteps", type=int, default=2)

    def get_parser(self):
        return self.parser