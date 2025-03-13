import os.path as osp
import math
import pickle
import warnings

import glob

import torch.utils.data as data
import torch.nn.functional as F
from torchvision.datasets.video_utils import VideoClips
from converter import  normalize, normalize_spectrogram, get_mel_spectrogram_from_audio
from torchaudio import transforms as Ta
from torchvision import transforms as Tv
from torchvision.io.video import read_video
import torch
from torchvision.transforms import InterpolationMode

class LatentDataset(data.Dataset):
    """ Generic dataset for latents pregenerated from a dataset
    Returns a dictionary of latents encoded from the original dataset """
    exts = ['pt']

    def __init__(self, data_folder, train=True):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
        """
        super().__init__()
        self.train = train

        folder = osp.join(data_folder, 'train' if train else 'test')
        self.files = sum([glob.glob(osp.join(folder, '**', f'*.{ext}'), recursive=True)
                     for ext in self.exts], [])

        warnings.filterwarnings('ignore')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        while True:
            try:
                latents = torch.load(self.files[idx], map_location="cpu")
            except Exception as e:
                print(f"Dataset Exception: {e}")
                idx = (idx + 1) % len(self.files)
                continue
            break

        return latents["video"], latents["audio"], latents["y"]
class AudioVideoDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm']

    def __init__(self, data_folder, train=True, resolution=64, sample_every_n_frames=1, sequence_length=8, audio_channels=1, sample_rate=16000, min_length=1, ignore_cache=False, labeled=True, target_video_fps=10):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution
        self.sample_every_n_frames = sample_every_n_frames
        self.audio_channels = audio_channels
        self.sample_rate = sample_rate
        self.min_length = min_length
        self.labeled = labeled


        folder = osp.join(data_folder, 'train' if train else 'test')
        files = sum([glob.glob(osp.join(folder, '**', f'*.{ext}'), recursive=True)
                     for ext in self.exts], [])

        # hacky way to compute # of classes (count # of unique parent directories)
        self.classes = list(set([get_parent_dir(f) for f in files]))
        self.classes.sort()
        self.class_to_label = {c: i for i, c in enumerate(self.classes)}

        warnings.filterwarnings('ignore')
        cache_file = osp.join(folder, f"metadata_{self.sequence_length}.pkl")
        if not osp.exists(cache_file) or ignore_cache or True:
            clips = VideoClips(files, self.sequence_length, num_workers=32, frame_rate=target_video_fps)
            # pickle.dump(clips.metadata, open(cache_file, 'wb'))
        else:
            metadata = pickle.load(open(cache_file, 'rb'))
            clips = VideoClips(files, self.sequence_length,
                               _precomputed_metadata=metadata)

        # self._clips = clips.subset(np.arange(24))
        self._clips = clips

    @property
    def n_classes(self):
        return len(self.classes)

    def __len__(self):
        return self._clips.num_clips()

    def __getitem__(self, idx):
        resolution = self.resolution
        while True:
            try:
                video, _, info, _ = self._clips.get_clip(idx)
            except Exception:
                idx = (idx + 1) % self._clips.num_clips()
                continue
            break

        return preprocess(video, resolution, sample_every_n_frames=self.sample_every_n_frames), self.get_audio(info, idx), self.get_label(idx)

    def get_label(self, idx):
        if not self.labeled:
            return -1
        video_idx, clip_idx = self._clips.get_clip_location(idx)
        class_name = get_parent_dir(self._clips.video_paths[video_idx])
        label = self.class_to_label[class_name]
        return label

    def get_audio(self, info, idx):
        video_idx, clip_idx = self._clips.get_clip_location(idx)

        video_path = self._clips.video_paths[video_idx]
        video_fps = self._clips.video_fps[video_idx]

        duration_per_frame = self._clips.video_pts[video_idx][1] - self._clips.video_pts[video_idx][0]
        clip_pts = self._clips.clips[video_idx][clip_idx]
        clip_pid = clip_pts // duration_per_frame

        start_t = (clip_pid[0] / video_fps * 1. ).item()
        end_t = ((clip_pid[-1] + 1) / video_fps * 1. ).item()

        _, raw_audio, _ = read_video(video_path,start_t, end_t, pts_unit='sec')
        raw_audio = prepare_audio(raw_audio, info["audio_fps"], self.sample_rate, self.audio_channels, self.sequence_length, self.min_length)

        _, spec = get_mel_spectrogram_from_audio(raw_audio[0].numpy())
        norm_spec = normalize_spectrogram(spec)
        norm_spec = normalize(norm_spec) # normalize to [-1, 1], because pipeline do not normalize for torch.Tensor input
        norm_spec.unsqueeze(1) # add channel dimension
        return norm_spec
        #return raw_audio[0]


def get_parent_dir(path):
    return osp.basename(osp.dirname(path))

def preprocess(video, resolution, sample_every_n_frames=1):
    video = video.permute(0, 3, 1, 2).float() / 255.  # TCHW
    
    old_size = video.shape[2:4]
    ratio = min(float(resolution)/(old_size[0]), float(resolution)/(old_size[1]) )
    new_size = tuple([int(i*ratio) for i in old_size])
    pad_w = resolution - new_size[1]
    pad_h = resolution- new_size[0]
    top,bottom = pad_h//2, pad_h-(pad_h//2)
    left,right = pad_w//2, pad_w -(pad_w//2)
    transform = Tv.Compose([Tv.Resize(new_size, interpolation=InterpolationMode.BICUBIC), Tv.Pad((left, top, right, bottom))])
    video_new = transform(video)

    video_new = video_new*2-1

    return video_new

def pad_crop_audio(audio, target_length):
    target_length = int(target_length)
    n, s = audio.shape
    start = 0
    end = start + target_length
    output = audio.new_zeros([n, target_length])
    output[:, :min(s, target_length)] = audio[:, start:end]
    return output

def prepare_audio(audio, in_sr, target_sr, target_channels, sequence_length, min_length):
    if in_sr != target_sr:
        resample_tf = Ta.Resample(in_sr, target_sr)
        audio = resample_tf(audio)

    max_length = target_sr/10*sequence_length
    target_length = max_length + (min_length - (max_length % min_length)) % min_length

    audio = pad_crop_audio(audio, target_length)

    audio = set_audio_channels(audio, target_channels)

    return audio

def set_audio_channels(audio, target_channels):
    if target_channels == 1:
        # Convert to mono
        # audio = audio.mean(0, keepdim=True)
        audio = audio[:1, :]
    elif target_channels == 2:
        # Convert to stereo
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2, :]
    return audio