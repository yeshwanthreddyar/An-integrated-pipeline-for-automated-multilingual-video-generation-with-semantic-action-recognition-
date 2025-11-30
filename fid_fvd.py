# fid_fvd.py
import os
import math
from typing import List, Tuple, Optional

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.io import read_video
from torchvision.models import inception_v3, Inception_V3_Weights
import torchvision.transforms as T
from torchvision.models.video import r3d_18

from scipy.linalg import sqrtm
from PIL import Image

# -------------------------------------------------------
# ⚙️ CONFIG
# -------------------------------------------------------
REAL_VIDEOS_DIR = "./UCF-101"      # Path to UCF-101 (or subset)
FAKE_VIDEOS_DIR = "./generated"    # Path to your generated videos
MAX_VIDEOS_REAL = 50               # How many real videos to sample
MAX_VIDEOS_FAKE = 50               # How many fake videos to sample
FRAMES_PER_VIDEO_FID = 8           # frames per video for FID
CLIP_LEN_FVD = 16                  # frames per video clip for FVD
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------
# 🧹 UTILITIES
# -------------------------------------------------------
def list_video_files(folder: str, max_videos: Optional[int] = None) -> List[str]:
    exts = (".mp4", ".avi", ".mov", ".mkv")
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    paths.sort()
    if max_videos is not None:
        paths = paths[:max_videos]
    return paths


def safe_read_video(path: str):
    """
    Use torchvision.io.read_video.
    Returns frames: [T, H, W, C] (uint8), or None on error.
    """
    try:
        video, _, _ = read_video(path, pts_unit="sec")
        # video: [T, H, W, C], uint8
        if video.ndim != 4 or video.size(0) == 0:
            raise ValueError("Empty or invalid video tensor")
        return video
    except Exception as e:
        print(f"⚠️ Skipping {path} due to error: {e}")
        return None


def sample_indices(num_frames: int, num_samples: int) -> np.ndarray:
    if num_frames <= 0:
        return np.array([], dtype=int)
    if num_frames <= num_samples:
        return np.arange(num_frames, dtype=int)
    return np.linspace(0, num_frames - 1, num_samples).astype(int)


# -------------------------------------------------------
# 🖼 FRAME EXTRACTION FOR FID
# -------------------------------------------------------
def extract_frames_for_fid(
    folder: str,
    max_videos: Optional[int] = None,
    frames_per_video: int = 8,
) -> List[np.ndarray]:
    """
    Returns a list of frames (as HxWx3 uint8 numpy arrays).
    These are used as 'images' for standard FID.
    """
    video_paths = list_video_files(folder, max_videos)
    print(f"📥 Decoding videos from {folder}: {len(video_paths)} files")

    all_frames: List[np.ndarray] = []

    for vp in tqdm(video_paths, desc="📥 Decoding videos for FID"):
        video = safe_read_video(vp)
        if video is None:
            continue

        # video: [T, H, W, C] uint8
        T_total = video.size(0)
        idx = sample_indices(T_total, frames_per_video)
        frames = video[idx]  # [k, H, W, C]

        for f in frames:
            frame_np = f.numpy()
            all_frames.append(frame_np)

    print(f"📥 Extracted {len(all_frames)} frames from {folder} for FID")
    return all_frames


# -------------------------------------------------------
# 🎥 CLIP EXTRACTION FOR FVD
# -------------------------------------------------------
def extract_clips_for_fvd(
    folder: str,
    max_videos: Optional[int] = None,
    clip_len: int = 16,
    resize_hw: Tuple[int, int] = (112, 112),
) -> List[torch.Tensor]:
    """
    Extracts fixed-length clips [C, T, H, W] from videos for FVD.
    Uses R3D-18-compatible resolution (112x112).
    """
    video_paths = list_video_files(folder, max_videos)
    print(f"📥 Decoding videos from {folder}: {len(video_paths)} files")

    clips: List[torch.Tensor] = []
    resize_h, resize_w = resize_hw

    for vp in tqdm(video_paths, desc="📥 Decoding videos for FVD"):
        video = safe_read_video(vp)
        if video is None:
            continue

        # video: [T, H, W, C] uint8
        T_total = video.size(0)
        if T_total == 0:
            continue

        # Sample indices for a single clip
        idx = sample_indices(T_total, clip_len)
        frames = video[idx]  # [k, H, W, C]

        # If fewer than clip_len, pad with last frame
        if frames.size(0) < clip_len:
            pad_num = clip_len - frames.size(0)
            pad_frame = frames[-1:].repeat(pad_num, 1, 1, 1)
            frames = torch.cat([frames, pad_frame], dim=0)

        # Resize and convert to [C, T, H, W], float32 in [0,1]
        frames = frames.permute(0, 3, 1, 2).float() / 255.0  # [T, C, H, W]
        frames = F.interpolate(
            frames.permute(1, 0, 2, 3).unsqueeze(0),  # [1, C, T, H, W]
            size=(clip_len, resize_h, resize_w),
            mode="trilinear",
            align_corners=False,
        )
        # frames: [1, C, T, H, W]
        clips.append(frames.squeeze(0))

    print(f"📥 Extracted {len(clips)} clips from {folder} for FVD")
    return clips


# -------------------------------------------------------
# 🧠 INCEPTION FEATURES FOR FID
# -------------------------------------------------------
def build_inception_model():
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights, transform_input=False)
    model.fc = nn.Identity()  # output 2048-dim
    model.eval().to(DEVICE)

    # Preprocessing transform for images to feed into Inception
    preprocess = weights.transforms()
    return model, preprocess


def frames_to_inception_features(
    frames: List[np.ndarray],
    model,
    preprocess,
    batch_size: int = 32,
) -> np.ndarray:
    feats_list: List[np.ndarray] = []

    for i in tqdm(range(0, len(frames), batch_size), desc="🧩 Extracting Inception features"):
        batch = frames[i : i + batch_size]
        if not batch:
            continue

        imgs = []
        for frame in batch:
            # frame: HxWx3 uint8
            img = Image.fromarray(frame)
            img = preprocess(img)  # [3, 299, 299]
            imgs.append(img)

        tensor_batch = torch.stack(imgs, dim=0).to(DEVICE)  # [B, 3, 299, 299]

        with torch.no_grad():
            out = model(tensor_batch)
            # Handle possible InceptionOutputs type
            if hasattr(out, "logits"):
                feat = out.logits
            else:
                feat = out
            feat = feat.cpu().numpy()
            feats_list.append(feat)

    if not feats_list:
        raise RuntimeError("No features extracted for FID (no valid frames).")

    feats = np.concatenate(feats_list, axis=0)  # [N, 2048]
    return feats


def calculate_fid(mu1, sigma1, mu2, sigma2) -> float:
    """
    Standard FID formula using scipy.linalg.sqrtm.
    """
    diff = mu1 - mu2
    diff_sq = diff.dot(diff)

    # sqrt of cov product
    covmean = sqrtm(sigma1.dot(sigma2))

    # Numerical issues => complex numbers; take real part
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff_sq + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


def compute_fid(real_dir: str, fake_dir: str) -> float:
    print("📊 Computing FID...")

    # 1) Extract frames
    real_frames = extract_frames_for_fid(
        real_dir,
        max_videos=MAX_VIDEOS_REAL,
        frames_per_video=FRAMES_PER_VIDEO_FID,
    )
    fake_frames = extract_frames_for_fid(
        fake_dir,
        max_videos=MAX_VIDEOS_FAKE,
        frames_per_video=FRAMES_PER_VIDEO_FID,
    )

    if len(real_frames) == 0 or len(fake_frames) == 0:
        raise RuntimeError("Not enough frames to compute FID.")

    # 2) Inception features
    inception, preprocess = build_inception_model()
    real_feats = frames_to_inception_features(real_frames, inception, preprocess)
    fake_feats = frames_to_inception_features(fake_frames, inception, preprocess)

    # 3) Means + covariances
    mu_r = np.mean(real_feats, axis=0)
    sigma_r = np.cov(real_feats, rowvar=False)

    mu_f = np.mean(fake_feats, axis=0)
    sigma_f = np.cov(fake_feats, rowvar=False)

    # 4) FID
    fid = calculate_fid(mu_r, sigma_r, mu_f, sigma_f)
    return fid


# -------------------------------------------------------
# 🎥 R3D-18 FEATURES FOR FVD
# -------------------------------------------------------
class R3DFeatureExtractor(nn.Module):
    """
    R3D-18 backbone up to penultimate layer.
    Output: [B, 512] feature vectors for each clip.
    """

    def __init__(self):
        super().__init__()
        # Load weights pretrained on Kinetics-400
        self.backbone = r3d_18(weights="KINETICS400_V1")
        self.backbone.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T, H, W]
        return: [B, 512]
        """
        return self.backbone(x)


def clips_to_r3d_features(
    clips: List[torch.Tensor],
    model: nn.Module,
    batch_size: int = 4,
) -> np.ndarray:
    feats_list: List[np.ndarray] = []

    for i in tqdm(range(0, len(clips), batch_size), desc="🎥 Extracting R3D features (FVD)"):
        batch_clips = clips[i : i + batch_size]
        if not batch_clips:
            continue

        x = torch.stack(batch_clips, dim=0).to(DEVICE)  # [B, C, T, H, W]
        with torch.no_grad():
            feat = model(x)  # [B, 512]
        feats_list.append(feat.cpu().numpy())

    if not feats_list:
        raise RuntimeError("No features extracted for FVD (no valid clips).")

    feats = np.concatenate(feats_list, axis=0)  # [N, 512]
    return feats


def compute_fvd(real_dir: str, fake_dir: str) -> float:
    """
    FVD-style metric using R3D-18 features instead of I3D.
    Mathematically same as FID but on video features.
    """
    print("📊 Computing FVD (R3D-18 feature space)...")

    real_clips = extract_clips_for_fvd(
        real_dir,
        max_videos=MAX_VIDEOS_REAL,
        clip_len=CLIP_LEN_FVD,
        resize_hw=(112, 112),
    )
    fake_clips = extract_clips_for_fvd(
        fake_dir,
        max_videos=MAX_VIDEOS_FAKE,
        clip_len=CLIP_LEN_FVD,
        resize_hw=(112, 112),
    )

    if len(real_clips) == 0 or len(fake_clips) == 0:
        raise RuntimeError("Not enough clips to compute FVD.")

    model = R3DFeatureExtractor().to(DEVICE).eval()

    real_feats = clips_to_r3d_features(real_clips, model)
    fake_feats = clips_to_r3d_features(fake_clips, model)

    mu_r = np.mean(real_feats, axis=0)
    sigma_r = np.cov(real_feats, rowvar=False)

    mu_f = np.mean(fake_feats, axis=0)
    sigma_f = np.cov(fake_feats, rowvar=False)

    fvd = calculate_fid(mu_r, sigma_r, mu_f, sigma_f)
    return fvd


# -------------------------------------------------------
# 🚀 MAIN
# -------------------------------------------------------
def main():
    print(f"💻 Using device: {DEVICE}")

    # FID
    try:
        fid = compute_fid(REAL_VIDEOS_DIR, FAKE_VIDEOS_DIR)
        print(f"\n✅ FID between {REAL_VIDEOS_DIR} and {FAKE_VIDEOS_DIR}: {fid:.2f}")
    except Exception as e:
        print(f"\n❌ Error while computing FID: {e}")

    # FVD
    try:
        fvd = compute_fvd(REAL_VIDEOS_DIR, FAKE_VIDEOS_DIR)
        print(f"\n✅ FVD (R3D-18) between {REAL_VIDEOS_DIR} and {FAKE_VIDEOS_DIR}: {fvd:.2f}")
    except Exception as e:
        print(f"\n❌ Error while computing FVD: {e}")


if __name__ == "__main__":
    main()
