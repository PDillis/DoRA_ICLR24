"""
Video dataset classes for DoRA pretraining.

Class hierarchy
---------------
DoRAVideoDataset            – abstract base
  SingleVideoDataset        – one long MP4 (Walking-Tours, StyleGAN single-walk, …)
  MultiVideoDataset         – directory of MP4s (multiple walks, CARLA clips, …)
    CARLADataset            – stub, raises NotImplementedError (implement when ready)
    RealWorldDrivingDataset – stub, raises NotImplementedError (implement when ready)

Backward-compat alias:  WT_dataset_1vid = SingleVideoDataset

Factory:  build_dataset(data_format, data_path, num_frames, step_between_clips, transform)
"""

import abc
import glob
import os

import decord
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Decord helper
# ---------------------------------------------------------------------------

class DecordInit:
    """Open a video file with Decord for CPU-side decoding."""

    def __init__(self, num_threads: int = 1):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)

    def __call__(self, filename: str) -> decord.VideoReader:
        return decord.VideoReader(filename, ctx=self.ctx, num_threads=self.num_threads)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class DoRAVideoDataset(torch.utils.data.Dataset, abc.ABC):
    """Abstract base class for DoRA video datasets.

    Subclasses must implement ``__len__`` and ``_load_clip``.
    ``__getitem__`` applies ``self.transform`` and converts to a Tensor.
    """

    def __init__(self, num_frames: int, step_between_clips: int, transform=None):
        self.num_frames = num_frames
        self.step_between_clips = step_between_clips
        self.transform = transform
        # Total number of raw frames spanned by one clip
        self._clip_len = num_frames * step_between_clips

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def _load_clip(self, index: int) -> np.ndarray:
        """Return a uint8 array of shape (T, H, W, C) for the clip at *index*."""
        ...

    def __getitem__(self, index):
        video = self._load_clip(index)
        with torch.no_grad():
            video = torch.from_numpy(video)
            if self.transform is not None:
                video = self.transform(video)
        return video


# ---------------------------------------------------------------------------
# Single-video dataset  (Walking-Tours, StyleGAN single-walk, …)
# ---------------------------------------------------------------------------

class SingleVideoDataset(DoRAVideoDataset):
    """Sample clips from a single MP4 video file.

    Works for any long-form video: Walking-Tours recordings, StyleGAN
    latent-walk exports, or any other continuous MP4.

    Args:
        video_path:         Path to the MP4 file.
        num_frames:         Frames per clip (temporal depth).
        step_between_clips: Stride between sampled frames within each clip.
        transform:          Optional callable applied to the raw (T,H,W,C) tensor.
    """

    def __init__(
        self,
        video_path: str,
        num_frames: int,
        step_between_clips: int,
        transform=None,
    ):
        super().__init__(num_frames, step_between_clips, transform)
        self.path = video_path
        self._decoder = DecordInit()
        self.total_frames = len(self._decoder(self.path))

    def __len__(self) -> int:
        return max(0, self.total_frames - self._clip_len)

    def _load_clip(self, index: int) -> np.ndarray:
        while True:
            try:
                reader = self._decoder(self.path)
                indices = np.arange(
                    index, index + self._clip_len, self.step_between_clips, dtype=int
                )
                frames = reader.get_batch(indices).asnumpy()
                del reader
                return frames
            except Exception as exc:
                print(exc)


# ---------------------------------------------------------------------------
# Multi-video dataset  (directory of MP4s)
# ---------------------------------------------------------------------------

class MultiVideoDataset(DoRAVideoDataset):
    """Sample clips from multiple MP4 files in a directory.

    Clips never cross video boundaries. Every MP4 in *video_dir* contributes
    an independent pool of valid clip-start positions.

    Args:
        video_dir:          Path to a directory. MP4s are found recursively.
        num_frames:         Frames per clip.
        step_between_clips: Stride between sampled frames within each clip.
        transform:          Optional callable applied to each clip.
        extensions:         File extensions to include (default: .mp4 / .MP4).
    """

    def __init__(
        self,
        video_dir: str,
        num_frames: int,
        step_between_clips: int,
        transform=None,
        extensions: tuple = ('.mp4', '.MP4'),
    ):
        super().__init__(num_frames, step_between_clips, transform)
        self._decoder = DecordInit()

        # Collect all video paths
        self._video_paths: list[str] = sorted(
            p
            for ext in extensions
            for p in glob.glob(os.path.join(video_dir, '**', f'*{ext}'), recursive=True)
        )
        if not self._video_paths:
            raise FileNotFoundError(f"No video files found under {video_dir!r}")

        # Per-video clip counts + cumulative index for O(log n) global→local mapping
        counts: list[int] = []
        for path in self._video_paths:
            try:
                n = len(self._decoder(path))
                counts.append(max(0, n - self._clip_len))
            except Exception as exc:
                print(f"Warning: skipping {path}: {exc}")
                counts.append(0)
        self._cum_counts = np.cumsum([0] + counts)

    def __len__(self) -> int:
        return int(self._cum_counts[-1])

    def _locate(self, index: int) -> tuple[int, int]:
        """Map a global index → (video_idx, local_frame_start)."""
        video_idx = int(np.searchsorted(self._cum_counts[1:], index, side='right'))
        local_idx = index - int(self._cum_counts[video_idx])
        return video_idx, local_idx

    def _load_clip(self, index: int) -> np.ndarray:
        video_idx, local_idx = self._locate(index)
        path = self._video_paths[video_idx]
        while True:
            try:
                reader = self._decoder(path)
                indices = np.arange(
                    local_idx, local_idx + self._clip_len, self.step_between_clips, dtype=int
                )
                frames = reader.get_batch(indices).asnumpy()
                del reader
                return frames
            except Exception as exc:
                print(exc)


# ---------------------------------------------------------------------------
# Domain-specific stubs  (implement when ready)
# ---------------------------------------------------------------------------

class CARLADataset(MultiVideoDataset):
    """CARLA simulator dataset — **not yet implemented**.

    When ready: collect CARLA autopilot recordings as MP4 files, point
    ``data_path`` at the directory, and implement any CARLA-specific
    pre-processing here (e.g. filtering out episodes with poor coverage,
    balancing weather conditions, etc.).
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "CARLADataset is not yet implemented. "
            "Export CARLA episodes as MP4 files and use MultiVideoDataset directly, "
            "or add CARLA-specific pre-processing in this class."
        )


class RealWorldDrivingDataset(MultiVideoDataset):
    """Real-world driving footage — **not yet implemented**.

    Intended for BDD100K, nuScenes, Waymo Open Dataset, KITTI, etc.
    When ready: download and extract the target dataset, export clips as MP4,
    point ``data_path`` at the clip directory, and implement any dataset-
    specific clip filtering / balancing here.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "RealWorldDrivingDataset is not yet implemented. "
            "Extract driving clips as MP4 files and use MultiVideoDataset directly, "
            "or add dataset-specific logic in this class."
        )


# ---------------------------------------------------------------------------
# Backward-compat alias
# ---------------------------------------------------------------------------

WT_dataset_1vid = SingleVideoDataset


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_dataset(
    data_format: str,
    data_path: str,
    num_frames: int,
    step_between_clips: int,
    transform=None,
) -> DoRAVideoDataset:
    """Instantiate the appropriate dataset class.

    ``data_format`` choices
    -----------------------
    ``'single_video'``          SingleVideoDataset  (one long MP4)
    ``'multi_video'``           MultiVideoDataset   (directory of MP4s)
    ``'stylegan'``              auto-detect: file → Single, directory → Multi
    ``'carla'``                 CARLADataset        (raises NotImplementedError)
    ``'bdd100k'`` /             RealWorldDrivingDataset (raises NotImplementedError)
    ``'nuscenes'`` /
    ``'real_world_driving'``
    """
    kwargs = dict(
        num_frames=num_frames,
        step_between_clips=step_between_clips,
        transform=transform,
    )

    if data_format == 'single_video':
        return SingleVideoDataset(data_path, **kwargs)

    if data_format == 'multi_video':
        return MultiVideoDataset(data_path, **kwargs)

    if data_format == 'stylegan':
        if os.path.isfile(data_path):
            return SingleVideoDataset(data_path, **kwargs)
        if os.path.isdir(data_path):
            return MultiVideoDataset(data_path, **kwargs)
        raise FileNotFoundError(f"data_path not found: {data_path!r}")

    if data_format == 'carla':
        return CARLADataset(data_path, **kwargs)

    if data_format in ('bdd100k', 'nuscenes', 'real_world_driving'):
        return RealWorldDrivingDataset(data_path, **kwargs)

    raise ValueError(
        f"Unknown data_format {data_format!r}. "
        "Valid choices: single_video, multi_video, stylegan, carla, bdd100k, nuscenes, real_world_driving."
    )
