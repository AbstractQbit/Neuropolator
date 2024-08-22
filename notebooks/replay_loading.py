import os
import itertools
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import pandas as pd

from osrparse import Replay, ReplayEvent, ReplayEventOsu, GameMode
from typing import Iterator, List, Tuple, Dict, Any, Optional, Union, Iterable, Callable
from tqdm import tqdm

from multiprocessing import Pool


def enum_replay_folder(path, full_path=True):
    for f in os.listdir(path):
        if f.endswith(".osr"):
            if full_path:
                yield os.path.join(path, f)
            else:
                yield f


def read_replay(filename):
    with open(filename, 'br') as f:
        return Replay.from_file(f)


def gamemode_match(gamemode: GameMode = GameMode.STD):
    return lambda replay: replay.mode is gamemode


def fix_time_deltas(replay: Iterable[ReplayEventOsu]):
    """
    Filter out negative delta events at the start and replace 0 delta with a 1.
    """
    for event in replay:
        if event.time_delta > 0:
            yield event
        elif event.time_delta == 0:
            yield ReplayEventOsu(time_delta=1, x=event.x, y=event.y, keys=event.keys)


def collapse_stationary(replay: Iterable[ReplayEventOsu]):
    """
    Collapse all consecutive stationary events and put their time_delta sum into the following event.
    """
    last_x, last_y = 0, 0
    cumulative_delta = 0
    for event in replay:
        if event.x == last_x and event.y == last_y:
            cumulative_delta += event.time_delta
        else:
            yield ReplayEventOsu(
                time_delta=event.time_delta + cumulative_delta,
                x=last_x,
                y=last_y,
                keys=event.keys)
            last_x, last_y = event.x, event.y
            cumulative_delta = 0


def get_stroke_splits(replay: List[ReplayEventOsu], delta_threshold=25):
    """
    Split a replay into strokes using a time delta threshold.
    """
    split_points = [0]
    for i in range(len(replay)):
        if replay[i].time_delta > delta_threshold:
            split_points.append(i)
    split_points.append(len(replay)+1)
    splits = []
    for i in range(len(split_points)-1):
        splits.append(replay[split_points[i]:split_points[i+1]])
    return splits


def preprocess_stroke(stroke: Iterable[ReplayEventOsu]):
    """
    Preprocess a stroke into a format suitable for spline interpolation.
    """
    positions = np.array([[event.x, event.y] for event in stroke])
    # repeat the ends of the stroke for spline to reach them
    positions = np.pad(positions, ((1,), (0,)), mode='edge')

    deltas = np.array([event.time_delta for event in stroke])
    # first delta of a stroke might be big or negative
    deltas[0] = 16
    # deltas might be too small and cause kinks in the spline
    # TODO: this might be problematic because key presses sometimes cause
    # valid small deltas, and this will make velocity hitch on streams
    deltas = np.maximum(deltas, 8)
    # repeat, same as with positions
    deltas = np.pad(deltas, (1), mode='edge')
    # 0 to make cumsum start from 0
    deltas[0] = 0

    timings = np.cumsum(deltas)
    return timings, positions


def files_to_strokes(filenames: Iterable[str], min_length=30, split_threshold=25):
    replays = map(read_replay, filenames)
    replays = filter(gamemode_match(GameMode.STD), replays)
    replays = (replay.replay_data for replay in replays)
    replays = map(fix_time_deltas, replays)
    replays = map(collapse_stationary, replays)
    replays = map(list, replays)
    # strokes = map(lambda x: get_stroke_splits(x, split_threshold), replays)
    strokes = (get_stroke_splits(x, split_threshold) for x in replays)
    strokes = itertools.chain.from_iterable(strokes)
    strokes = filter(lambda x: len(x) > min_length, strokes)
    strokes = map(preprocess_stroke, strokes)
    return strokes


def cubic_barry_goldman_weights(knots, t):
    knots10 = knots[1] - knots[0]
    knots21 = knots[2] - knots[1]
    knots32 = knots[3] - knots[2]

    knots20 = knots[2] - knots[0]
    knots31 = knots[3] - knots[1]

    knots0t = knots[0] - t
    knots1t = knots[1] - t
    knots2t = knots[2] - t
    knots3t = knots[3] - t

    b1 = knots2t / knots21
    b2 = -knots1t / knots21

    a1 = b1 * knots2t / knots20
    a2 = b1 * -knots0t / knots20 + b2 * knots3t / knots31
    a3 = b2 * -knots1t / knots31

    p0 = a1 * knots1t / knots10
    p1 = a1 * -knots0t / knots10 + a2 * knots2t / knots21
    p2 = a2 * -knots1t / knots21 + a3 * knots3t / knots32
    p3 = a3 * -knots2t / knots32

    return np.array([p0, p1, p2, p3])


def sample_stroke(timings, positions, rate, offset=0, max_length=2048):
    full_start, full_end = timings[1] + offset, timings[-2] - 1
    full_duration = full_end - full_start
    max_duration = max_length * 1000 / rate
    if max_duration < full_duration:
        # If the maximum duration is less than the full duration, randomly select a start time within the range that allows for the maximum duration
        start = np.random.uniform(full_start, full_end - max_duration)
        end = start + max_duration
    else:
        start, end = full_start, full_end
    num_points = int((end - start) * rate / 1000)

    t_interp = np.linspace(start, end, num_points)
    t_ind = np.searchsorted(timings, t_interp, 'right').astype(int)
    knots_interp = np.take(timings, [t_ind-2, t_ind-1, t_ind, t_ind+1], axis=0, mode='clip')
    knot_positions = np.take(positions, [t_ind-2, t_ind-1, t_ind, t_ind+1], axis=0)
    knot_weights = cubic_barry_goldman_weights(knots_interp, t_interp)
    interp_positions = np.sum(knot_positions * knot_weights[:, :, None], axis=0)

    return interp_positions


@torch.jit.script
def cubic_barry_goldman_weights_torch(knots, t):
    knots10 = knots[1] - knots[0]
    knots21 = knots[2] - knots[1]
    knots32 = knots[3] - knots[2]

    knots20 = knots[2] - knots[0]
    knots31 = knots[3] - knots[1]

    knots0t = knots[0] - t
    knots1t = knots[1] - t
    knots2t = knots[2] - t
    knots3t = knots[3] - t

    b1 = knots2t / knots21
    b2 = -knots1t / knots21

    a1 = b1 * knots2t / knots20
    a2 = b1 * -knots0t / knots20 + b2 * knots3t / knots31
    a3 = b2 * -knots1t / knots31

    p0 = a1 * knots1t / knots10
    p1 = a1 * -knots0t / knots10 + a2 * knots2t / knots21
    p2 = a2 * -knots1t / knots21 + a3 * knots3t / knots32
    p3 = a3 * -knots2t / knots32

    return torch.stack([p0, p1, p2, p3])

# todo fix this
# @torch.jit.script
# def sample_stroke_torch(timings, positions, rate, offset=0, max_length=2048):
#     full_start, full_end = timings[1] + offset, timings[-2] - 1
#     full_duration = full_end - full_start
#     max_duration = max_length * 1000 / rate
#     if max_duration < full_duration:
#         # If the maximum duration is less than the full duration, randomly select a start time within the range that allows for the maximum duration
#         start = torch.rand(1) * (full_end - max_duration) + full_start
#         end = start + max_duration
#     else:
#         start, end = full_start, full_end
#     num_points = int((end - start) * rate / 1000)

#     t_interp = torch.linspace(start, end, num_points)
#     t_ind = torch.searchsorted(timings, t_interp, right=True)
#     knots_interp = torch.take(timings, torch.stack([t_ind-2, t_ind-1, t_ind, t_ind+1]), dim=0, mode='clip')
#     knot_positions = torch.take(positions, torch.stack([t_ind-2, t_ind-1, t_ind, t_ind+1]), dim=0)
#     knot_weights = cubic_barry_goldman_weights_torch(knots_interp, t_interp)
#     interp_positions = torch.sum(knot_positions * knot_weights.unsqueeze(2), dim=0)

#     return interp_positions


if __name__ == "__main__":
    # replay_fns = list(enum_replays("Replays/"))
    replay_fns = list(itertools.islice(enum_replay_folder("H:/osu!/Data/r/"), 200))
    # print(*replay_fns, sep="\n")

    strokes = files_to_strokes(replay_fns)

    stroke_lengths = map(len, strokes)
    # stroke_lengths = filter(lambda x: x > 30, stroke_lengths)

    stroke_lengths = list(tqdm(stroke_lengths))

    # with Pool(16) as p:
    #     stroke_lengths = list(tqdm(p.imap(len, strokes, 1024)))

    slp = pd.Series(stroke_lengths).quantile(np.linspace(0, 1, 11))
    print(slp)

    # with Pool(8) as p:
    #     replay_data = list(tqdm(p.imap(read_replay_data, replay_fns, 64), total=len(replay_fns)))
    # replay_data = [read_replay_data(fn) for fn in tqdm(replay_fns)]




