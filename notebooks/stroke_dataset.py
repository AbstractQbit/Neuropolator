import random
import numpy as np
from scipy import stats as st
from scipy.ndimage import gaussian_filter1d

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler

from multiprocessing import Manager

from replay_loading import sample_stroke


class StrokeDataset(Dataset):
    def __init__(
        self,
        strokes,
        common_transforms=None,
        input_transforms=None,
        target_transforms=None,
        common_split_transforms=None,
    ):
        m = Manager()
        self.strokes = m.list(strokes)
        self.common_transforms = common_transforms
        self.input_transforms = input_transforms
        self.target_transforms = target_transforms
        self.common_split_transforms = common_split_transforms
        self.wrand_sampler = WeightedRandomSampler([len(s[0]) for s in strokes], len(strokes), replacement=True)

    def __len__(self):
        return len(self.strokes)

    def __getitem__(self, idx):
        sample = self.strokes[idx]

        for transform in self.common_transforms:
            sample = transform(sample)
        target = sample.copy()

        for transform in self.input_transforms:
            sample = transform(sample)

        for transform in self.target_transforms:
            target = transform(target)

        if self.common_split_transforms:
            for transform in self.common_split_transforms:
                sample = transform(sample)
                target = transform(target)

        return sample, target


class StrokeResample:
    def __init__(self, rate_dist=st.uniform(30, 250), max_length=2048):
        self.rate_dist = rate_dist
        self.max_length = max_length

    def __call__(self, sample):
        timings, positions = sample
        rate = self.rate_dist.rvs(1).item()
        offset = np.random.uniform(0, 1 / rate)
        return sample_stroke(timings, positions, rate, offset, max_length=self.max_length)


class ScaleRotateFlip:
    def __init__(self, scale_dist=st.uniform(0.5, 1.5)):
        self.scale_dist = scale_dist

    def __call__(self, sample):
        scale = self.scale_dist.rvs(1).item()
        sample = sample * scale
        angle = random.uniform(-np.pi, np.pi)
        flip = random.choice([1, -1])
        rotation_matrix = np.array([[np.cos(angle), -flip * np.sin(angle)], [flip * np.sin(angle), np.cos(angle)]])
        sample = sample @ rotation_matrix
        return sample


class LeftPad:
    def __init__(self, target_length=2048):
        self.target_length = target_length

    def __call__(self, sample):
        padding_width = self.target_length - len(sample)
        return np.pad(sample, ((padding_width, 0), (0, 0)), mode="constant", constant_values=0)


class AddGaussianNoise:
    def __init__(self, std_dist=st.expon(scale=0.5)):
        self.std_dist = std_dist

    def __call__(self, sample):
        std = self.std_dist.rvs(1).item()
        noise = np.random.normal(0, std, sample.shape)
        return sample + noise


class SmoothWithGaussian:
    def __init__(self, sigma_dist=st.expon(scale=0.5)):
        self.sigma_dist = sigma_dist

    def __call__(self, sample):
        sigma = self.sigma_dist.rvs(1).item()
        return gaussian_filter1d(sample, sigma, axis=0)


class StrokeDiff:
    def __call__(self, sample):
        return np.diff(sample, axis=0)


class StrokeToTensor:
    def __call__(self, sample):
        return torch.from_numpy(sample).float()


def collate_simple_stack(batch):
    return torch.stack([stroke for stroke, _ in batch]).mT, torch.stack([target for _, target in batch]).mT


def default_transforms(seq_len):
    return {
        "common_transforms": [
            StrokeResample(max_length=seq_len),
            ScaleRotateFlip(),
            LeftPad(target_length=seq_len),
        ],
        "input_transforms": [
            AddGaussianNoise(),
        ],
        "target_transforms": [
            SmoothWithGaussian(),
        ],
        "common_split_transforms": [
            StrokeDiff(),
            StrokeToTensor(),
        ],
    }
